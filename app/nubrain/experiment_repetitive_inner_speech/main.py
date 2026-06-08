import multiprocessing as mp
import os
import traceback
from time import sleep

import pygame

from nubrain.audio.tone import generate_tone
from nubrain.device.device_interface import create_eeg_device
from nubrain.experiment_image.randomize_conditions import (
    create_balanced_list,
    sample_next_image,
    sample_with_min_distance,
    shuffle_dicts_with_repetitions,
)
from nubrain.experiment_repetitive_inner_speech.data import eeg_data_logging
from nubrain.experiment_repetitive_inner_speech.load_experiment_config import (
    load_config_repetitive_inner_speech_yaml,
)
from nubrain.global_config import GlobalConfig
from nubrain.image.tools import get_all_images, load_and_scale_image
from nubrain.misc.datetime import get_formatted_current_datetime

# mp.set_start_method("spawn", force=True)  # Necessary on if running on windows?

yaml_file_path = "/home/john/github/nubrain-connect/app/nubrain/experiment_repetitive_inner_speech/example_experiment_config.yaml"
config = load_config_repetitive_inner_speech_yaml(yaml_file_path=yaml_file_path)
config["subject_id"] = "sub-999"
config["session_id"] = "1"


def experiment_image(config: dict):
    # ----------------------------------------------------------------------------------
    # *** Get config

    device_type = config["device_type"]
    lsl_stream_name = config["lsl_stream_name"]

    subject_id = config["subject_id"]
    session_id = config["session_id"]

    utility_frequency = config["utility_frequency"]

    output_directory = config["output_directory"]
    path_stimuli = config["path_stimuli"]

    storage_bucket_name = config["storage_bucket_name"]
    storage_blob_name = config["storage_blob_name"]
    storage_bucket_credentials = config["storage_bucket_credentials"]

    stimulus_duration = config["stimulus_duration"]
    post_stim_interval = config["post_stim_interval"]
    n_repetitions_per_trial = config["n_repetitions_per_trial"]
    cue_duration = config["cue_duration"]
    repeat_duration = config["repeat_duration"]
    inter_trial_interval = config["inter_trial_interval"]
    inter_trial_jitter = config["inter_trial_jitter"]
    n_trials_per_block = config["n_trials_per_block"]
    inter_block_rest_duration = config["inter_block_rest_duration"]
    n_blocks = config["n_blocks"]
    n_target_events = config["n_target_events"]
    response_window_duration = config["response_window_duration"]

    stimulus_font_sizes = config["stimulus_font_sizes"]
    stimulus_font_color = config["stimulus_font_color"]
    background_color = config["background_color"]

    audio_cue_frequency = config["audio_cue_frequency"]
    audio_cue_duration = config["audio_cue_duration"]
    audio_cue_amplitude = config["audio_cue_amplitude"]

    eeg_channel_mapping = config.get("eeg_channel_mapping", None)
    eeg_device_address = config.get("eeg_device_address", None)

    global_config = GlobalConfig()

    # ----------------------------------------------------------------------------------
    # *** Test if output path exists

    if not os.path.isdir(output_directory):
        raise AssertionError(f"Target directory does not exist: {output_directory}")

    current_datetime = get_formatted_current_datetime()
    path_out_data = os.path.join(output_directory, f"eeg_{current_datetime}.h5")

    if os.path.isfile(path_out_data):
        raise AssertionError(f"Target file already exists: {path_out_data}")

    # ----------------------------------------------------------------------------------
    # *** Get input images & their categories

    images_and_categories = get_all_images(image_directory=path_stimuli)

    if not images_and_categories:
        raise AssertionError(f"Found no images at {path_stimuli}")
    print(f"Found {len(images_and_categories)} images")

    # images_and_categories = [
    #     {
    #         "image_file_path": "/path/to/image.png",
    #         "image_category": "horse",
    #     },
    # ]

    # ----------------------------------------------------------------------------------
    # *** Create pseudo-random condition order

    n_trials = n_blocks * n_trials_per_block

    # Ensure number of trials is even (requirement for splitting trials evenly in text
    # and image trials).
    if not ((n_trials % 2) == 0):
        print(
            f"Adjusting number of trials from {n_trials} to {n_trials + 1} "
            "(required for splitting into equal number of text and image trials)"
        )
        n_trials += 1

    # List with all unique image categories (e.g. `["apple", "banana", ...]`).
    image_categories = list(set([x["image_category"] for x in images_and_categories]))

    # We need at least twice as many trials as categories (for one text and one image
    # trial per category).
    n_image_categories = len(image_categories)

    if n_trials < (n_image_categories * 2):
        print(
            f"Adjusting number of trials from {n_trials} to {n_image_categories * 2} "
            "(required for having one text & image trial each per category)"
        )
        n_trials = n_image_categories * 2

    stimulus_categories = []
    for stimulus_class in image_categories:
        for stimulus_type in ["text", "image"]:
            stimulus_categories.append(
                {
                    "stimulus_class": stimulus_class,
                    "stimulus_type": stimulus_type,
                }
            )

    # Order of image categories.
    trial_order = create_balanced_list(
        image_categories=stimulus_categories,
        target_length=n_trials,
    )

    # Pseudo-random trial order (no repetitions).
    trial_order = shuffle_dicts_with_repetitions(
        list_with_duplicates=trial_order,
        repetitions=0,
    )

    # Mapping from image categories to image file paths, e.g. `{"apple":
    # ["/path/to/apple_1.png", "/path/to/apple_2.png", ...], "banana":
    # ["/path/to/banana_2.png", ...]}`.
    category_to_filepath = {}
    for item in images_and_categories:
        image_category = item["image_category"]
        image_filepath = item["image_file_path"]
        if image_category in category_to_filepath:
            category_to_filepath[image_category].append(image_filepath)
        else:
            category_to_filepath[image_category] = [image_filepath]

    previous_image_file_path = None
    previous_image_category = None

    # ----------------------------------------------------------------------------------
    # *** Create target events

    # Indices of target events.
    target_trial_idcs = sample_with_min_distance(
        n_samples=n_target_events,
        lower=10,  # No targets at very beginning
        upper=(n_trials - 10),  # No targets at the very end
        min_distance=1,
    )

    # ----------------------------------------------------------------------------------
    # *** Prepare EEG measurement

    print(f"Initializing EEG device: {device_type}")

    device_kwargs = {"eeg_channel_mapping": eeg_channel_mapping}
    if device_type in ["cyton", "synthetic"]:
        device_kwargs["eeg_device_address"] = eeg_device_address
    elif device_type == "dsi24":
        device_kwargs["lsl_stream_name"] = lsl_stream_name
    else:
        raise ValueError(f"Unexpected `device_type`: {device_type}")

    eeg_device = create_eeg_device(device_type, **device_kwargs)

    eeg_device.prepare_session()

    # This is a bit clunky. At this point, `eeg_channel_mapping` is None or a dict with
    # a channel mapping from the config yaml file. Overwrite it with the channel mapping
    # from the device (in case of the DSI-24 device, the channel mapping from the device
    # is used in any case).
    eeg_channel_mapping = eeg_device.eeg_channel_mapping

    # Need to start the stream before calling `eeg_device.get_device_info()`, because
    # we retrieve data from board to determine data shape (number of channels).
    eeg_device.start_stream()
    sleep(0.1)

    # Get device info.
    device_info = eeg_device.get_device_info()
    eeg_board_description = device_info["board_description"]
    eeg_sampling_rate = device_info["sampling_rate"]
    eeg_channels = device_info["eeg_channels"]
    marker_channel = device_info["marker_channel"]
    n_channels_total = device_info["n_channels_total"]

    if device_type in ["cyton", "synthetic"]:
        # For Cyton device, we need to get the number of EEG channels from the device
        # (not sure, this might only work after starting the stream).
        eeg_device.eeg_channels = eeg_channels
        eeg_device.timestamp_channel = eeg_board_description["timestamp_channel"]

    print(f"Board: {eeg_board_description['name']}")
    print(f"Sampling Rate: {eeg_sampling_rate} Hz")
    print(f"EEG Channels: {eeg_channels}")
    print(f"Marker Channel: {marker_channel}")
    print(f"EEG Channel Mapping: {eeg_channel_mapping}")

    board_data, board_timestamps = eeg_device.get_board_data()

    print(f"Board data dtype: {board_data.dtype}")
    print(f"Board data shape: {board_data.shape}")
    print(f"Board timestamps shape: {board_timestamps.shape}")

    # ----------------------------------------------------------------------------------
    # *** Start data logging subprocess

    data_logging_queue = mp.Queue()

    subprocess_params = {
        # EEG parameters
        "device_type": device_type,
        "lsl_stream_name": lsl_stream_name,
        "utility_frequency": utility_frequency,
        "eeg_channel_mapping": eeg_channel_mapping,
        "eeg_device_address": eeg_device_address,
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
        "n_trials_per_block": n_trials_per_block,
        "inter_block_rest_duration": inter_block_rest_duration,
        "n_blocks": n_blocks,
        "n_target_events": n_target_events,
        "response_window_duration": response_window_duration,
        # Storage
        "output_directory": output_directory,
        "path_stimuli": path_stimuli,
        "storage_bucket_name": storage_bucket_name,
        "storage_blob_name": storage_blob_name,
        "storage_bucket_credentials": storage_bucket_credentials,
        # Stimulus properties
        "stimulus_font_sizes": stimulus_font_sizes,
        "stimulus_font_color": stimulus_font_color,
        "background_color": background_color,
        "audio_cue_frequency": audio_cue_frequency,
        "audio_cue_duration": audio_cue_duration,
        "audio_cue_amplitude": audio_cue_amplitude,
    }

    logging_process = mp.Process(target=eeg_data_logging, args=(subprocess_params,))
    logging_process.daemon = True
    logging_process.start()

    # ----------------------------------------------------------------------------------
    # *** Start experiment

    # Performance counters.
    n_correct_answers = 0

    running = True
    while running:
        pygame.init()

        # ------------------------------------------------------------------------------
        # *** Prepare audio cues

        # Use an audio cue for the start of the repetitive inner speech period.
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)

        # Get the sample rate from the mixer settings.
        sample_rate = pygame.mixer.get_init()[0]

        audio_cue_tone_data = generate_tone(
            frequency=audio_cue_frequency,  # Pitch of the tone in Hz
            duration=audio_cue_duration,  # Duration of audio cue
            amplitude=audio_cue_amplitude,  # Volume, from 0.0 to 1.0
            sample_rate=sample_rate,
        )

        # Create a sound object from the numpy array.
        audio_cue_tone = pygame.sndarray.make_sound(audio_cue_tone_data)

        # ------------------------------------------------------------------------------
        # *** Prepare visual stimulus generation

        # Get screen dimensions and set up full screen.
        screen_info = pygame.display.Info()
        screen_width = screen_info.current_w
        screen_height = screen_info.current_h
        screen = pygame.display.set_mode(
            (screen_width, screen_height), pygame.FULLSCREEN
        )
        pygame.display.set_caption("Image Presentation Experiment")
        pygame.mouse.set_visible(False)

        try:
            # Initial grey screen.
            pygame.time.wait(100)
            screen.fill(background_color)
            pygame.display.flip()
            pygame.time.wait(100)
            screen.fill(background_color)
            pygame.display.flip()

            # Clear board buffer.
            _, _ = eeg_device.get_board_data()

            # Pause for specified number of milliseconds.
            pygame.time.delay(int(round(inter_block_rest_duration * 1000.0)))

            # Loop over trials.
            for idx_trial in range(n_trials):
                if not running:  # Check for quit event
                    break

                # ------------------------------------------------------------------
                # *** (1) Stimulus presentation (image or word)

                # E.g. "apple".
                next_category = trial_order[idx_trial]["stimulus_class"]
                # "text" or "image".
                next_type = trial_order[idx_trial]["stimulus_type"]

                # Sample the next image.
                next_file_path = sample_next_image(
                    next_image_category=next_category,
                    category_to_filepath=category_to_filepath,
                    previous_image_file_path=previous_image_file_path,
                )

                # Load the next image.
                image_and_metadata = load_and_scale_image(
                    image_file_path=next_file_path,
                    screen_width=screen_width,
                    screen_height=screen_height,
                )
                if image_and_metadata is None:
                    raise AssertionError(f"Failed to load stimulus: {next_file_path}")

                current_image = image_and_metadata["image"]

                img_rect = current_image.get_rect(
                    center=(screen_width // 2, screen_height // 2)
                )
                screen.fill(background_color)
                screen.blit(current_image, img_rect)
                pygame.display.flip()
                # Start of stimulus presentation.
                t_stim_start = eeg_device.lsl_local_clock()

                # When using an OpenBCI device, we insert a stimulus marker into the
                # time series data on the EEG board. These markers can be used
                # during analysis to identify stimulus events. For the DSI-24
                # device, we instead use LSL timestamps stored in the hdf5 file for
                # identifying stimulus events.
                if device_type in ["cyton", "synthetic"]:
                    eeg_device.insert_marker(global_config.stim_start_marker)
                else:
                    data_logging_queue.put(
                        {
                            "type": "marker",
                            "marker_value": global_config.stim_start_marker,
                            "timestamp": t_stim_start,
                        }
                    )

                # Send pre-stimulus EEG data (to avoid buffer overflow).
                eeg_data, eeg_ts = eeg_device.get_board_data()
                if eeg_data.size > 0:
                    data_logging_queue.put(
                        {
                            "type": "eeg",
                            "eeg_data": eeg_data,
                            "eeg_timestamps": eeg_ts,
                        }
                    )

                stimulus_data = {
                    "stimulus_start_time": t_stim_start,
                    "stimulus_class": next_category,  # E.g. "apple"
                    "stimulus_type": next_type,  # "text" or "image"
                }

                # Wait for image duration, but check for responses continuously.
                t_stim_end_expected = t_stim_start + stimulus_duration
                while eeg_device.lsl_local_clock() < t_stim_end_expected:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            running = False
                        if event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_ESCAPE:
                                running = False
                    if not running:
                        break
                if not running:
                    break

                # ------------------------------------------------------------------
                # *** (2) Post-stimulus delay

                # End of stimulus presentation. Display empty screen.
                screen.fill(background_color)
                pygame.display.flip()
                t_stim_end_actual = eeg_device.lsl_local_clock()

                stimulus_data["stimulus_end_time"] = t_stim_end_actual
                stimulus_data["stimulus_duration_s"] = (
                    t_stim_end_actual - stimulus_data["stimulus_start_time"]
                )

                if device_type in ["cyton", "synthetic"]:
                    eeg_device.insert_marker(global_config.stim_end_marker)
                else:
                    data_logging_queue.put(
                        {
                            "type": "marker",
                            "marker_value": global_config.stim_end_marker,
                            "timestamp": t_stim_end_actual,
                        }
                    )

                # Send EEG data from stimulus presentation period.
                eeg_data, eeg_ts = eeg_device.get_board_data()
                if eeg_data.size > 0:
                    data_logging_queue.put(
                        {
                            "type": "eeg",
                            "eeg_data": eeg_data,
                            "eeg_timestamps": eeg_ts,
                        }
                    )

                # Time until when to show empty screen (post-stimulus delay).
                t_isi_end = t_stim_end_actual + post_stim_interval

                # Continue checking for quit events.
                t_stim_end_expected = t_stim_start + stimulus_duration
                while eeg_device.lsl_local_clock() < t_stim_end_expected:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            running = False
                        if event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_ESCAPE:
                                running = False
                    if not running:
                        break
                if not running:
                    break

                # ------------------------------------------------------------------
                # *** (3) Silent speech

                stimulus_data["silent_speech_cue_onsets"] = []

                for idx_repetition in range(n_repetitions_per_trial):
                    # Play the tone that cues the beginning of an inner speech
                    # repetition. Note that `.play()` is non-blocking.
                    audio_cue_tone.play()

                    # Because `.play()` is non-blocking, we only log the onset of
                    # the audio cue (and not the end).
                    t_cue_start = eeg_device.lsl_local_clock()

                    stimulus_data["silent_speech_cue_onsets"].append(t_cue_start)

                    if device_type in ["cyton", "synthetic"]:
                        eeg_device.insert_marker(global_config.cue_start_marker)
                    else:
                        data_logging_queue.put(
                            {
                                "type": "marker",
                                "marker_value": global_config.cue_start_marker,
                                "timestamp": t_cue_start,
                            }
                        )

                    # Send EEG data from silent speech period.
                    eeg_data, eeg_ts = eeg_device.get_board_data()
                    if eeg_data.size > 0:
                        data_logging_queue.put(
                            {
                                "type": "eeg",
                                "eeg_data": eeg_data,
                                "eeg_timestamps": eeg_ts,
                            }
                        )

                    # Continue checking for quit events.
                    t_repeat_end_expected = t_cue_start + repeat_duration
                    while eeg_device.lsl_local_clock() < t_repeat_end_expected:
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                running = False
                            if event.type == pygame.KEYDOWN:
                                if event.key == pygame.K_ESCAPE:
                                    running = False
                        if not running:
                            break
                    if not running:
                        break

                # ------------------------------------------------------------------
                # *** (4) Attention task

                # Is this a target trial?
                if idx_trial in target_trial_idcs:
                    raise NotImplementedError

                # ------------------------------------------------------------------
                # *** (5) Inter-stimulus interval

                inter_trial_interval
                inter_trial_jitter
                # + np.random.uniform(low=0.0, high=inter_trial_jitter)

                # ----------------------------------------------------------------------
                # *** (6) Inter-block interval

                if ((idx_trial + 1) % n_trials_per_block) == 0:
                    # TODO: Break
                    raise NotImplementedError

            running = False

            # Send final board data.
            eeg_data, eeg_ts = eeg_device.get_board_data()
            if eeg_data.size > 0:
                data_logging_queue.put(
                    {"type": "eeg", "eeg_data": eeg_data, "eeg_timestamps": eeg_ts}
                )

        except Exception as e:
            print(f"An error occurred during the experiment: {e}")
            print(traceback.format_exc())
            running = False
        finally:
            pygame.quit()
            print("Experiment closed.")

    eeg_device.stop_stream()
    eeg_device.release_session()

    print("Join process for sending data")
    data_logging_queue.put(None)
    logging_process.join()
