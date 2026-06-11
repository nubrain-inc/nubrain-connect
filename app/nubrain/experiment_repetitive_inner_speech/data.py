import json
import os
from time import time

import h5py
import numpy as np

from nubrain.global_config import GlobalConfig
from nubrain.storage.gcloud_bucket_upload import upload_to_gcs


def eeg_data_logging(subprocess_params: dict):
    """
    Log experimental data.

    Continuously save to local hdf file. Upload to google cloud storage bucket at the
    end of the run. To be run in separate process (using multiprocessing).

    Please note that the stimulus onset and offset timestamps (`stimulus_start_time` and
    `stimulus_end_time`) use the LSL local clock, which is in seconds, but not aligned
    with UNIX epoch. We use the LSL clock for consistency with EEG timestamps from the
    DSI-24 device. Only the `experiment_start_time` is in UNIX epoch.
    """
    # ----------------------------------------------------------------------------------
    # *** Get parameters

    global_config = GlobalConfig()

    # EEG parameters
    device_type = subprocess_params["device_type"]
    lsl_stream_name = subprocess_params["lsl_stream_name"]
    utility_frequency = subprocess_params["utility_frequency"]
    eeg_board_description = subprocess_params["eeg_board_description"]
    eeg_sampling_rate = subprocess_params["eeg_sampling_rate"]
    n_channels_total = subprocess_params["n_channels_total"]
    eeg_channel_mapping = subprocess_params["eeg_channel_mapping"]
    # Parameters not used by DSI-24, for compatibility with Cyton board
    eeg_device_address = subprocess_params["eeg_device_address"]
    eeg_channels = subprocess_params["eeg_channels"]
    marker_channel = subprocess_params["marker_channel"]
    # Session parameters
    subject_id = subprocess_params["subject_id"]
    session_id = subprocess_params["session_id"]
    # Experiment structure / timing
    trial_order = subprocess_params["trial_order"]
    target_trial_idcs = subprocess_params["target_trial_idcs"]
    stimulus_duration = subprocess_params["stimulus_duration"]
    post_stim_interval = subprocess_params["post_stim_interval"]
    n_repetitions_per_trial = subprocess_params["n_repetitions_per_trial"]
    cue_duration = subprocess_params["cue_duration"]
    repeat_duration = subprocess_params["repeat_duration"]
    inter_trial_interval = subprocess_params["inter_trial_interval"]
    inter_trial_jitter = subprocess_params["inter_trial_jitter"]
    n_trials = subprocess_params["n_trials"]
    n_trials_per_block = subprocess_params["n_trials_per_block"]
    inter_block_rest_duration = subprocess_params["inter_block_rest_duration"]
    n_blocks = subprocess_params["n_blocks"]
    n_target_events = subprocess_params["n_target_events"]
    # Storage
    path_out_data = subprocess_params["path_out_data"]
    path_stimuli = subprocess_params["path_stimuli"]
    storage_bucket_name = subprocess_params["storage_bucket_name"]
    storage_blob_name = subprocess_params["storage_blob_name"]
    storage_bucket_credentials = subprocess_params["storage_bucket_credentials"]
    # Stimulus properties
    stimulus_font_name = subprocess_params["stimulus_font_name"]
    stimulus_font_is_bold = subprocess_params["stimulus_font_is_bold"]
    stimulus_font_is_italic = subprocess_params["stimulus_font_is_italic"]
    stimulus_font_size = subprocess_params["stimulus_font_size"]
    stimulus_font_spacing = subprocess_params["stimulus_font_spacing"]
    stimulus_font_color = subprocess_params["stimulus_font_color"]
    background_color = subprocess_params["background_color"]
    audio_cue_frequency = subprocess_params["audio_cue_frequency"]
    audio_cue_duration = subprocess_params["audio_cue_duration"]
    audio_cue_amplitude = subprocess_params["audio_cue_amplitude"]

    data_logging_queue = subprocess_params["data_logging_queue"]

    # ----------------------------------------------------------------------------------
    # *** Create and initialize HDF5 file

    experiment_metadata = {
        "config_version": global_config.config_version,
        "stim_start_marker": global_config.stim_start_marker,
        "stim_end_marker": global_config.stim_end_marker,
        "cue_start_marker": global_config.cue_start_marker,
        "hdf5_dtype": global_config.hdf5_dtype,
        "experiment_start_time": time(),  # Epoch timestamp
        # EEG parameters
        "device_type": device_type,
        "lsl_stream_name": lsl_stream_name,
        "utility_frequency": utility_frequency,
        "eeg_board_description": eeg_board_description,
        "eeg_sampling_rate": eeg_sampling_rate,
        "n_channels_total": n_channels_total,
        "eeg_channel_mapping": eeg_channel_mapping,
        # Parameters not used by DSI-24, for compatibility with Cyton board
        "eeg_device_address": eeg_device_address,
        "eeg_channels": eeg_channels,
        "marker_channel": marker_channel,
        # Session parameters
        "subject_id": subject_id,
        "session_id": session_id,
        # Experiment structure / timing
        "trial_order": trial_order,
        "target_trial_idcs": target_trial_idcs,
        "stimulus_duration": stimulus_duration,
        "post_stim_interval": post_stim_interval,
        "n_repetitions_per_trial": n_repetitions_per_trial,
        "cue_duration": cue_duration,
        "repeat_duration": repeat_duration,
        "inter_trial_interval": inter_trial_interval,
        "inter_trial_jitter": inter_trial_jitter,
        "n_trials": n_trials,
        "n_trials_per_block": n_trials_per_block,
        "inter_block_rest_duration": inter_block_rest_duration,
        "n_blocks": n_blocks,
        "n_target_events": n_target_events,
        # Storage
        "path_out_data": path_out_data,
        "path_stimuli": path_stimuli,
        "storage_bucket_name": storage_bucket_name,
        "storage_blob_name": storage_blob_name,
        # "storage_bucket_credentials": storage_bucket_credentials,
        # Stimulus properties
        "stimulus_font_name": stimulus_font_name,
        "stimulus_font_is_bold": stimulus_font_is_bold,
        "stimulus_font_is_italic": stimulus_font_is_italic,
        "stimulus_font_size": stimulus_font_size,
        "stimulus_font_spacing": stimulus_font_spacing,
        "stimulus_font_color": stimulus_font_color,
        "background_color": background_color,
        "audio_cue_frequency": audio_cue_frequency,
        "audio_cue_duration": audio_cue_duration,
        "audio_cue_amplitude": audio_cue_amplitude,
    }

    print(f"Initializing HDF5 file at: {path_out_data}")
    with h5py.File(path_out_data, "w") as file:
        # ------------------------------------------------------------------------------
        # *** Initialize hdf5 dataset for metadata

        # Create group for metadata.
        metadata_group = file.create_group("metadata")

        # Iterate over the Python dictionary and save each item as an attribute of the
        # "metadata" group.
        for key, value in experiment_metadata.items():
            # HDF5 attributes have limitations on data types. Complex types like
            # dictionaries or tuples are not natively supported. We check if the value
            # is a type that needs to be converted to a string. JSON is a convenient
            # format for this serialization.
            if isinstance(value, (dict, list, tuple)):
                # Serialize the complex type into a JSON string.
                metadata_group.attrs[key] = json.dumps(value)
            elif value is None:
                # Skip None values (not supported by hdf5, would result in TypeError).
                pass
            else:
                metadata_group.attrs[key] = value

        # ------------------------------------------------------------------------------
        # *** Initialize hdf5 dataset for EEG data

        # Initialize dataset for EEG and additional channels. To handle a variable
        # number of timesteps, create a resizable dataset. We specify an initial shape
        # but set the 'maxshape' to allow one of the dimensions to be unlimited (by
        # setting it to None). 'chunks=True' is recommended for resizable datasets for
        # better performance. It lets h5py decide the chunk size.

        file.create_dataset(
            "eeg_data",
            shape=(n_channels_total, 0),
            maxshape=(n_channels_total, None),  # fixed_channels, unlimited_timesteps
            dtype=global_config.hdf5_dtype,
            chunks=True,
        )

        file.create_dataset(
            "eeg_timestamps",
            shape=(0,),
            maxshape=(None,),
            dtype="float64",
            chunks=True,
        )

        file.create_dataset(
            "marker_data",
            shape=(2, 0),  # timestamp, marker value
            maxshape=(2, None),
            dtype="float64",
            chunks=True,
        )

        # ------------------------------------------------------------------------------
        # *** Initialize hdf5 datasets for stimulus and target data

        # Main stimulus dataset (common to all trials).
        stimulus_dtype = np.dtype(
            [
                ("idx_trial", np.int64),
                ("stimulus_class", h5py.string_dtype(encoding="utf-8")),
                ("stimulus_type", h5py.string_dtype(encoding="utf-8")),
                ("image_file_path", h5py.string_dtype(encoding="utf-8")),
                ("stimulus_start_time", np.float64),
                ("stimulus_end_time", np.float64),
                ("is_target", bool),
            ]
            + [
                (f"silent_speech_cue_onset_{x}", np.float64)
                for x in range(n_repetitions_per_trial)
            ]
        )

        file.create_dataset(
            "stimulus_data",
            (n_trials,),
            dtype=stimulus_dtype,
        )

        # Target dataset (specific to target events).
        targets_dtype = np.dtype(
            [
                ("idx_trial", np.int64),  # Foreign key to link back to stimulus_data
                ("attention_task_question", h5py.string_dtype(encoding="utf-8")),
                ("attention_task_answer_options", h5py.string_dtype(encoding="utf-8")),
                ("attention_task_selected_answer_idx", np.int64),
                ("attention_task_is_correct", bool),
                ("attention_task_response_time", np.float64),
            ]
        )

        file.create_dataset(
            "targets",
            (n_target_events,),
            dtype=targets_dtype,
        )

        # ------------------------------------------------------------------------------
        # *** Initialize hdf5 dataset for behavioural data

        behavioural_dtype = np.dtype(
            [
                ("n_target_events_correct", np.int64),
                ("n_target_events_incorrect", np.int64),
            ]
        )

        file.create_dataset(
            "behavioural_data",
            (1,),
            dtype=behavioural_dtype,
        )

    # ----------------------------------------------------------------------------------
    # *** Experiment loop

    stimulus_counter = 0
    target_counter = 0

    while True:
        new_data = data_logging_queue.get(block=True)

        if new_data is None:
            # Received None. End process.
            print("Ending preprocessing & data saving process.")
            break

        data_type = new_data["type"]

        with h5py.File(path_out_data, "a") as file:
            # --------------------------------------------------------------------------
            # *** Write EEG data to hdf5 file

            if data_type == "eeg":
                new_eeg_data = new_data.get("eeg_data")
                new_timestamps = new_data.get("eeg_timestamps")

                if new_eeg_data is not None and new_eeg_data.size > 0:
                    # Write EEG data.
                    hdf5_eeg_data = file["eeg_data"]
                    n_existing = hdf5_eeg_data.shape[1]
                    n_new = new_eeg_data.shape[1]
                    hdf5_eeg_data.resize(n_existing + n_new, axis=1)
                    hdf5_eeg_data[:, n_existing:] = new_eeg_data

                    # Write EEG timestamps.
                    hdf5_timestamps = file["eeg_timestamps"]
                    n_existing_ts = hdf5_timestamps.shape[0]
                    hdf5_timestamps.resize(n_existing_ts + n_new, axis=0)
                    hdf5_timestamps[n_existing_ts:] = new_timestamps

            # --------------------------------------------------------------------------
            # *** Write stimulus markers to hdf5 file

            elif data_type == "marker":
                marker_value = new_data.get("marker_value")
                marker_timestamp = new_data.get("timestamp")

                if marker_value is not None:
                    hdf5_marker_data = file["marker_data"]
                    n_existing = hdf5_marker_data.shape[1]
                    hdf5_marker_data.resize(n_existing + 1, axis=1)
                    hdf5_marker_data[:, n_existing] = (marker_timestamp, marker_value)

            # --------------------------------------------------------------------------
            # *** Write stimulus and target data to hdf5 file

            elif data_type == "stimulus":
                new_stimulus_data = new_data.get("stimulus_data")

                if new_stimulus_data is not None:
                    # Write to main stimulus dataset.
                    hdf5_stimulus_data = file["stimulus_data"]
                    data_to_write_stim = np.empty((1,), dtype=stimulus_dtype)

                    data_to_write_stim[0]["idx_trial"] = new_stimulus_data["idx_trial"]
                    data_to_write_stim[0]["stimulus_class"] = (
                        new_stimulus_data.get("stimulus_class") or ""
                    )
                    data_to_write_stim[0]["stimulus_type"] = (
                        new_stimulus_data.get("stimulus_type") or ""
                    )
                    data_to_write_stim[0]["image_file_path"] = (
                        new_stimulus_data.get("image_file_path") or ""
                    )
                    data_to_write_stim[0]["stimulus_start_time"] = new_stimulus_data[
                        "stimulus_start_time"
                    ]
                    data_to_write_stim[0]["stimulus_end_time"] = new_stimulus_data[
                        "stimulus_end_time"
                    ]

                    silent_speech_cue_onsets = new_stimulus_data[
                        "silent_speech_cue_onsets"
                    ]
                    for i in range(n_repetitions_per_trial):
                        data_to_write_stim[0][f"silent_speech_cue_onset_{i}"] = (
                            silent_speech_cue_onsets[i]
                        )

                    is_target = new_stimulus_data["is_target"]
                    data_to_write_stim[0]["is_target"] = is_target

                    hdf5_stimulus_data[stimulus_counter] = data_to_write_stim
                    stimulus_counter += 1

                    # Write to targets dataset (only if the current trial is a target
                    # event).
                    if is_target:
                        hdf5_targets_data = file["targets"]
                        data_to_write_target = np.empty((1,), dtype=targets_dtype)

                        # Write relational key.
                        data_to_write_target[0]["idx_trial"] = new_stimulus_data[
                            "idx_trial"
                        ]

                        attention_task_log = new_stimulus_data["attention_task_log"]
                        # attention_task_log:
                        # {
                        #     "question": question_text,
                        #     "answers": answers,
                        #     "selected_answer_idx": selected_answer_idx,
                        #     "is_correct": is_correct,
                        #     "response_time": response_time,
                        # }

                        data_to_write_target[0]["attention_task_question"] = (
                            attention_task_log["question"] or ""
                        )
                        data_to_write_target[0]["attention_task_answer_options"] = (
                            json.dumps(attention_task_log["answers"] or "")
                        )
                        # Integer data type does not support nan value, use -1 instead.
                        data_to_write_target[0][
                            "attention_task_selected_answer_idx"
                        ] = attention_task_log["selected_answer_idx"] or -1
                        data_to_write_target[0]["attention_task_is_correct"] = (
                            attention_task_log["is_correct"] or False
                        )
                        data_to_write_target[0]["attention_task_response_time"] = (
                            attention_task_log["response_time"] or np.nan
                        )

                        hdf5_targets_data[target_counter] = data_to_write_target
                        target_counter += 1

            # --------------------------------------------------------------------------
            # *** Write behavioural data to hdf5 file

            elif data_type == "behavioural":
                new_behavioural_data = new_data.get("behavioural_data")

                if new_behavioural_data is not None:
                    hdf5_behavioural_data = file["behavioural_data"]

                    n_correct = new_behavioural_data["n_target_events_correct"]
                    n_incorrect = new_behavioural_data["n_target_events_incorrect"]

                    data_to_write = np.empty((1,), dtype=behavioural_dtype)
                    data_to_write[0]["n_target_events_correct"] = n_correct
                    data_to_write[0]["n_target_events_incorrect"] = n_incorrect

                    # Write the structured array to the dataset.
                    hdf5_behavioural_data[0] = data_to_write

    # ----------------------------------------------------------------------------------
    # *** Upload to cloud storage

    # Upload hdf5 file to google cloud storage bucket at the end of the run.

    filename = os.path.split(path_out_data)[-1]

    _storage_blob_name = storage_blob_name.format(
        device_type=device_type,
        filename=filename,
    )

    upload_to_gcs(
        local_file_path=path_out_data,
        bucket_name=storage_bucket_name,
        destination_blob_name=_storage_blob_name,
        credentials_file_path=storage_bucket_credentials,
    )
