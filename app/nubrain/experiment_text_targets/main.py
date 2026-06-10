"""
Data collection mode with text stimuli. Word repetitions are target events for attention
task.
"""

import json
import multiprocessing as mp
import os
import random
import traceback
from copy import deepcopy
from time import sleep

import numpy as np
import pygame

from nubrain.audio.tone import generate_tone
from nubrain.device.device_interface import create_eeg_device
from nubrain.experiment_common.io import ExperimentIO
from nubrain.experiment_text_targets.behavioural_results import show_behavioural_results
from nubrain.experiment_text_targets.data import eeg_data_logging
from nubrain.experiment_text_targets.text_config import TextConfig
from nubrain.misc.datetime import get_formatted_current_datetime
from nubrain.text.rendering import construct_fonts, render_spaced_text

mp.set_start_method("spawn", force=True)  # Necessary on if running on windows?


def experiment_text_targets(config: dict):
    # ----------------------------------------------------------------------------------
    # *** Get config

    device_type = config["device_type"]
    lsl_stream_name = config.get("lsl_stream_name", "DSI-24")

    subject_id = config["subject_id"]
    session_id = config["session_id"]

    output_directory = config["output_directory"]
    path_stimuli = config["path_stimuli"]

    storage_bucket_name = config["storage_bucket_name"]
    storage_blob_name = config["storage_blob_name"]
    storage_bucket_credentials = config["storage_bucket_credentials"]

    eeg_channel_mapping = config.get("eeg_channel_mapping", None)

    utility_frequency = config["utility_frequency"]

    initial_rest_duration = config["initial_rest_duration"]
    stimulus_duration = config["stimulus_duration"]
    stimulus_jitter = config["stimulus_jitter"]
    stimulus_extension_target = config["stimulus_extension_target"]
    isi_duration = config["isi_duration"]
    isi_jitter = config["isi_jitter"]
    isi_extension_target = config["isi_extension_target"]
    inter_block_rest_duration = config["inter_block_rest_duration"]
    n_chars_long_word_threshold = config["n_chars_long_word_threshold"]
    extra_duration_per_char = config["extra_duration_per_char"]
    max_extra_stimulus_duration = config["max_extra_stimulus_duration"]

    section_idx_start = config["section_idx_start"]
    n_sections_to_show = config["n_sections_to_show"]

    stimuli_per_block = config["stimuli_per_block"]
    stimulus_font_sizes = config["stimulus_font_sizes"]
    stimulus_font_min_spacing = config["stimulus_font_min_spacing"]
    stimulus_font_max_spacing = config["stimulus_font_max_spacing"]

    eeg_device_address = config.get("eeg_device_address", None)

    text_config = TextConfig()

    # ----------------------------------------------------------------------------------
    # *** Test if output path exists

    if not os.path.isdir(output_directory):
        raise AssertionError(f"Target directory does not exist: {output_directory}")

    current_datetime = get_formatted_current_datetime()
    path_out_data = os.path.join(output_directory, f"eeg_{current_datetime}.h5")

    if os.path.isfile(path_out_data):
        raise AssertionError(f"Target file already exists: {path_out_data}")

    # ----------------------------------------------------------------------------------
    # *** Load stimulus data from JSON file

    with open(path_stimuli, "r", encoding="utf-8") as file:
        stimuli = json.load(file)

    text_sections = stimuli["text_sections"]

    # Only used for logging.
    min_distance_targets = stimuli["min_distance_targets"]
    min_words_per_section = stimuli["min_words_per_section"]
    ratio_target_events = stimuli["ratio_target_events"]
    words_per_section = stimuli["words_per_section"]

    # Select subset of text.
    text_sections = text_sections[
        section_idx_start : (section_idx_start + n_sections_to_show)
    ]

    text = [x for xs in [x["text_with_targets"] for x in text_sections] for x in xs]
    is_target = [x for xs in [x["is_target"] for x in text_sections] for x in xs]

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
        "device_type": device_type,
        "subject_id": subject_id,
        "session_id": session_id,
        "path_stimuli": path_stimuli,
        # EEG parameters
        "eeg_board_description": eeg_board_description,
        "eeg_sampling_rate": eeg_sampling_rate,
        "n_channels_total": n_channels_total,
        "eeg_channels": eeg_channels,
        "marker_channel": marker_channel,
        "eeg_channel_mapping": eeg_channel_mapping,
        "eeg_device_address": eeg_device_address,
        # Timing parameters
        "initial_rest_duration": initial_rest_duration,
        "stimulus_duration": stimulus_duration,
        "isi_duration": isi_duration,
        "isi_jitter": isi_jitter,
        "isi_extension_target": isi_extension_target,
        "inter_block_rest_duration": inter_block_rest_duration,
        "n_chars_long_word_threshold": n_chars_long_word_threshold,
        "extra_duration_per_char": extra_duration_per_char,
        "max_extra_stimulus_duration": max_extra_stimulus_duration,
        # Experiment structure
        "section_idx_start": section_idx_start,
        "n_sections_to_show": n_sections_to_show,
        "min_distance_targets": min_distance_targets,
        "min_words_per_section": min_words_per_section,
        "ratio_target_events": ratio_target_events,
        "words_per_section": words_per_section,
        "stimuli_per_block": stimuli_per_block,
        "stimulus_font_sizes": stimulus_font_sizes,
        "stimulus_font_min_spacing": stimulus_font_min_spacing,
        "stimulus_font_max_spacing": stimulus_font_max_spacing,
        # Text and targets
        "text": text,  # List of str
        "is_target": is_target,  # List of bool
        # Storage
        "path_out_data": path_out_data,
        "storage_bucket_name": storage_bucket_name,
        "storage_blob_name": storage_blob_name,
        "storage_bucket_credentials": storage_bucket_credentials,
        # Misc
        "utility_frequency": utility_frequency,
        "data_logging_queue": data_logging_queue,
    }

    logging_process = mp.Process(target=eeg_data_logging, args=(subprocess_params,))
    logging_process.daemon = True
    logging_process.start()

    # ----------------------------------------------------------------------------------
    # *** Set up shared I/O helper

    # Bundles eeg_device + device_type + data_logging_queue so the loop below can use
    # io.now(), io.wait_until(), io.emit_marker(), io.drain_eeg() without repeating the
    # plumbing (or the device-specific marker branch).
    io = ExperimentIO(
        eeg_device=eeg_device,
        device_type=device_type,
        data_logging_queue=data_logging_queue,
    )

    # ----------------------------------------------------------------------------------
    # *** Start experiment

    # Performance counters. A dict so the response handler (a closure passed to
    # io.wait_until via on_event) can update them in place across trials.
    counters = {"n_hits": 0, "n_false_alarms": 0}
    n_total_targets = sum(is_target)

    running = True

    pygame.init()

    # ----------------------------------------------------------------------------------
    # *** Prepare audio cues

    # Use an audio cue at the beginning and at the end of the inter-block interval,
    # so the participant can close their eyes / rest. These are rest cues only (not
    # analysis events), so their exact onset latency does not matter.

    # How long before the end of the inter-block interval to play the audio cue.
    pure_tone_end_delay = 1.0

    # Play the tone to cue the end of the inter-block interval x seconds before the
    # end of the inter-block interval. Do not use the audio cue if the inter-block
    # interval is too short.
    if inter_block_rest_duration <= (pure_tone_end_delay + 0.1):
        print(
            "WARNING: Will not use audio cue for the end of the inter-block "
            "interval because of short inter-block interval of "
            f"{inter_block_rest_duration} s"
        )
        use_ibi_audio_cue = False
    else:
        use_ibi_audio_cue = True

    if use_ibi_audio_cue:
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)

        # Get the sample rate from the mixer settings.
        sample_rate = pygame.mixer.get_init()[0]

        # Generate the tone data.
        tone_data_start = generate_tone(
            frequency=700,  # Pitch of the tone in Hz
            duration=0.3,  # Duration of audio cue
            amplitude=0.9,  # Volume, from 0.0 to 1.0
            sample_rate=sample_rate,
        )

        tone_data_end = generate_tone(
            frequency=1400,  # Pitch of the tone in Hz
            duration=0.3,  # Duration of audio cue
            amplitude=0.9,  # Volume, from 0.0 to 1.0
            sample_rate=sample_rate,
        )

        # Create a sound object from the numpy array.
        pure_tone_start = pygame.sndarray.make_sound(tone_data_start)
        pure_tone_end = pygame.sndarray.make_sound(tone_data_end)

    # ----------------------------------------------------------------------------------
    # *** Prepare visual stimulus generation

    # Get screen dimensions and set up full screen.
    screen_info = pygame.display.Info()
    screen_width = screen_info.current_w
    screen_height = screen_info.current_h
    screen = pygame.display.set_mode((screen_width, screen_height), pygame.FULLSCREEN)
    pygame.display.set_caption("Silent Reading Experiment")
    pygame.mouse.set_visible(False)

    try:
        # Initial grey screen.
        pygame.time.wait(100)
        screen.fill(text_config.rest_condition_color)
        pygame.display.flip()
        pygame.time.wait(100)
        screen.fill(text_config.rest_condition_color)
        pygame.display.flip()

        stimulus_fonts = construct_fonts(font_sizes=stimulus_font_sizes)

        # Clear board buffer.
        io.discard_eeg()

        # Pause for specified number of milliseconds.
        pygame.time.delay(int(round(initial_rest_duration * 1000.0)))

        # Send pre-stimulus EEG data (to avoid buffer overflow).
        io.drain_eeg()

        # Count stimuli to introduce breaks between blocks.
        stimulus_block_counter = 0
        t_stim_end_actual = None
        need_to_log_previous_stimulus = False

        # Per-trial response state. A dict so the response handler (a closure) can
        # mutate it in place. Importantly, this is reset after the "log previous
        # stimulus" block on each trial, so that block still records the *previous*
        # word's response. Do not move the reset earlier.
        response_state = {"made": False, "time": np.nan}

        # Loop through words.
        for word, is_target_event in zip(text, is_target):
            if not running:  # Check for quit event
                break

            # Extend stimulus duration for long words.
            n_chars = len(word)
            if n_chars > n_chars_long_word_threshold:
                # By how many characters does the current word exceed the character
                # threshold for extending stimulus duration.
                n_excess_chars = n_chars - n_chars_long_word_threshold
                extra_stimulus_duration = extra_duration_per_char * n_excess_chars
                # Never prolong stimulus duration for more than x seconds (irrespective
                # of number of characters).
                extra_stimulus_duration = min(
                    extra_stimulus_duration, max_extra_stimulus_duration
                )
            else:
                # Word length is not above threshold (regular stimulus duration).
                extra_stimulus_duration = 0.0

            # Randomly sample a font (we render the stimulus using different fonts to
            # achieve different stimulus appearance in terms of low-level visual
            # features.
            font_data = random.choice(stimulus_fonts)
            font_name = font_data["font_name"]
            font_size = font_data["font_size"]
            font_is_bold = font_data["font_is_bold"]
            font_is_italic = font_data["font_is_italic"]
            font_color = random.choice(text_config.font_colors)
            font_spacing = np.random.uniform(
                low=stimulus_font_min_spacing,
                high=stimulus_font_max_spacing,
            )

            # Clear previous stimulus.
            screen.fill(text_config.rest_condition_color)

            stimulus_text = render_spaced_text(
                text=word,
                font=font_data["font"],
                color=font_color,
                spacing=font_spacing,
            )

            stimulus_rect = stimulus_text.get_rect(
                center=(screen_width // 2, screen_height // 2)
            )
            screen.blit(stimulus_text, stimulus_rect)

            # --------------------------------------------------------------------------
            # *** Stimulus

            pygame.display.flip()
            # Start of stimulus presentation.
            t_stim_start = io.now()

            # --------------------------------------------------------------------------
            # *** Log previous stimulus

            # Now that we have flipped the screen and are showing the stimulus, take the
            # time and log data from previous stimulus. We do not need to log the
            # previous stimulus if there was an ISI or inter-block interval (in that
            # case, the stimulus gets logged at the beginning of the ISI or inter-block
            # interval, because the beginning of that interval determines the stimulus
            # end time).
            #
            # NOTE: `response_state["time"]` still holds the *previous* word's response
            # here, because the reset below has not run yet this trial.
            if need_to_log_previous_stimulus:
                if t_stim_end_actual is None:
                    # If there was no ISI or inter-block interval, the end time of the
                    # previous stimulus is determined by the onset of the current
                    # stimulus.
                    t_stim_end_actual = t_stim_start

                io.emit_marker(text_config.stim_end_marker, t_stim_end_actual)

                stimulus_data["response_time_s"] = response_state["time"]
                stimulus_data["stimulus_end_time"] = t_stim_end_actual
                stimulus_data["stimulus_duration_s"] = (
                    t_stim_end_actual - stimulus_data["stimulus_start_time"]
                )

                data_logging_queue.put(
                    {"type": "stimulus", "stimulus_data": stimulus_data}
                )

            # --------------------------------------------------------------------------
            # *** Continue stimulus presentation

            # When using an OpenBCI device, this inserts a hardware marker into the
            # board's time series; for the DSI-24 it queues an LSL-timestamped marker
            # instead. Both paths are handled by io.emit_marker().
            io.emit_marker(text_config.stim_start_marker, t_stim_start)

            if stimulus_jitter > 0.0:
                # Randomly sample stimulus duration jitter for the current trial.
                stimulus_jitter_current_trial = np.random.uniform(
                    low=0.0,
                    high=stimulus_jitter,
                )
            else:
                stimulus_jitter_current_trial = 0.0

            # Reset the response state for THIS trial (after logging the previous one
            # above). The deadline includes the target extensions so that a late press
            # during the (extended) ISI still counts.
            response_state["made"] = False
            response_state["time"] = np.nan
            response_deadline = (
                t_stim_start  # Stimulus start time
                + stimulus_duration  # Regular stimulus duration
                + extra_stimulus_duration  # Extra stimulus duration for long words
                + stimulus_extension_target  # Extra stimulus duration for targets
                + stimulus_jitter_current_trial  # Random stimulus duration jitter
                + isi_duration  # Inter stimulus interval (can be zero)
                + isi_extension_target  # Extra ISI for targets (can be zero)
            )

            def handle_response(event):
                """
                Spacebar response handler for the current trial (on_event hook).

                Called by io.wait_until for every non-quit event during the stimulus and
                ISI waits. Records the first in-window press and tallies it as a hit or
                false alarm.
                """
                if (
                    event.type == pygame.KEYDOWN
                    and event.key == pygame.K_SPACE
                    and not response_state["made"]
                ):
                    keydown_time = io.now()
                    if keydown_time < response_deadline:
                        response_state["made"] = True
                        response_state["time"] = keydown_time - t_stim_start
                        print(f"Response time: {round(response_state['time'], 3)}")
                        if is_target_event:
                            counters["n_hits"] += 1  # Hit.
                        else:
                            counters["n_false_alarms"] += 1  # False alarm.

            # Wait for stimulus duration.
            t_stim_end_expected = (
                t_stim_start  # Stimulus start time
                + stimulus_duration  # Regular stimulus duration
                + extra_stimulus_duration  # Extra stimulus duration for long words
                + stimulus_jitter_current_trial  # Random stimulus duration jitter
            )
            if is_target_event:
                # Extra stimulus duration for targets.
                t_stim_end_expected += stimulus_extension_target

            # The data from the current stimulus will be logged *after* flipping the
            # screen for the next stimulus. Keep a deepcopy so as to log the parameters
            # of the current stimulus (not the next one).
            stimulus_data = deepcopy(
                {
                    "stimulus_start_time": t_stim_start,
                    "word": word,
                    "font_name": font_name,
                    "font_size": font_size,
                    "font_is_bold": font_is_bold,
                    "font_is_italic": font_is_italic,
                    "font_spacing": font_spacing,
                    "font_color": font_color,
                    "is_target_event": is_target_event,
                }
            )
            need_to_log_previous_stimulus = True

            stimulus_block_counter += 1

            # Continue stimulus presentation until the current stimulus time is up,
            # collecting spacebar responses via the on_event handler.
            if io.wait_until(t_stim_end_expected, on_event=handle_response):
                running = False
                break

            # Log EEG data.
            io.drain_eeg()

            # --------------------------------------------------------------------------
            # *** Inter-block interval

            if stimulus_block_counter == stimuli_per_block:
                # Inter-block interval (break).
                screen.fill(text_config.rest_condition_color)
                pygame.display.flip()
                # Start of inter-block interval.
                t_ibi_start = io.now()

                if use_ibi_audio_cue:
                    # Audio cue to signal the beginning of the inter-block interval.
                    pure_tone_start.play()

                # ----------------------------------------------------------------------
                # *** Log previous stimulus

                # The end time of the previous stimulus is the onset of the current
                # inter-block interval. `response_state["time"]` holds this word's
                # response (collected during its stimulus wait above).
                t_stim_end_actual = t_ibi_start

                io.emit_marker(text_config.stim_end_marker, t_stim_end_actual)

                stimulus_data["response_time_s"] = response_state["time"]
                stimulus_data["stimulus_end_time"] = t_stim_end_actual
                stimulus_data["stimulus_duration_s"] = (
                    t_stim_end_actual - stimulus_data["stimulus_start_time"]
                )

                data_logging_queue.put(
                    {"type": "stimulus", "stimulus_data": stimulus_data}
                )

                need_to_log_previous_stimulus = False

                # ----------------------------------------------------------------------
                # *** Continue inter-block interval

                # End of inter-block interval.
                t_ibi_end = t_ibi_start + inter_block_rest_duration

                # Time when to play the cue signalling the end of the interval.
                t_ibi_end_audio_cue = None
                if use_ibi_audio_cue:
                    t_ibi_end_audio_cue = t_ibi_end - pure_tone_end_delay

                # Mutable state for the on_tick closure (so it can remember whether the
                # end-cue has already been played this interval).
                ibi_state = {"end_cue_played": False}

                def ibi_tick():
                    if (
                        use_ibi_audio_cue
                        and not ibi_state["end_cue_played"]
                        and t_ibi_end_audio_cue is not None
                        and io.now() >= t_ibi_end_audio_cue
                    ):
                        # Play the cue to signal the end of the inter-block interval.
                        pure_tone_end.play()
                        ibi_state["end_cue_played"] = True

                # Unlike the original, the inter-block wait now also pumps the event
                # queue (via io.wait_until), so the participant/experimenter can quit
                # during rest and the window stays responsive. (No response is collected
                # during the rest break, so no on_event handler here.)
                if io.wait_until(t_ibi_end, on_tick=ibi_tick):
                    running = False

                stimulus_block_counter = 0

                # Send inter-block EEG data (to avoid buffer overflow).
                io.drain_eeg()

                continue

            # --------------------------------------------------------------------------
            # *** ISI

            # Compute the duration of the upcoming inter stimulus interval (ISI). ISI
            # duration can be zero.
            next_isi_duration = isi_duration
            if is_target_event:
                # If this is a target event, prolong the ISI duration, to allow the
                # subject to respond before the onset of the next stimulus, to reduce
                # the probability of a motor response artefact in the subsequent trial.
                next_isi_duration += isi_extension_target
            if isi_jitter > 0.0:
                # Randomly sample ISI duration jitter for the current trial.
                isi_jitter_current_trial = np.random.uniform(
                    low=0.0,
                    high=isi_jitter,
                )
                next_isi_duration += isi_jitter_current_trial

            # The ISI interval can be zero; in that case, do not include an ISI at all.
            if next_isi_duration < 0.0167:
                # Skip ISI if ISI duration is less than one frame (assuming 60 Hz
                # refresh rate). The stimulus stays on screen for now.
                print("Skipping ISI")
                t_stim_end_actual = None  # No ISI, the stimulus is still shown
                continue

            # End of stimulus presentation. Display ISI grey screen.
            screen.fill(text_config.rest_condition_color)
            pygame.display.flip()
            t_stim_end_actual = io.now()
            # Time until when to show grey screen (ISI).
            t_isi_end = t_stim_end_actual + next_isi_duration

            # Continue checking for late responses or quit events. The same
            # handle_response (and the same response_state) is reused, so a press
            # already made during the stimulus is not double-counted.
            if io.wait_until(t_isi_end, on_event=handle_response):
                running = False
                break

            # Send post-stimulus EEG data (to avoid buffer overflow).
            io.drain_eeg()

        # ------------------------------------------------------------------------------
        # *** Log final stimulus data

        if need_to_log_previous_stimulus:
            if t_stim_end_actual is None:
                screen.fill(text_config.rest_condition_color)
                pygame.display.flip()
                t_stim_end_actual = io.now()

            io.emit_marker(text_config.stim_end_marker, t_stim_end_actual)

            stimulus_data["response_time_s"] = response_state["time"]
            stimulus_data["stimulus_end_time"] = t_stim_end_actual
            stimulus_data["stimulus_duration_s"] = (
                t_stim_end_actual - stimulus_data["stimulus_start_time"]
            )

            data_logging_queue.put({"type": "stimulus", "stimulus_data": stimulus_data})

        # ------------------------------------------------------------------------------
        # *** Show behavioural results

        # End of word loop. Calculate behavioural results.
        n_hits = counters["n_hits"]
        n_false_alarms = counters["n_false_alarms"]
        n_misses = n_total_targets - n_hits

        # Write behavioural results to hdf5 file.
        behavioural_data = {
            "n_total_targets": n_total_targets,
            "n_hits": n_hits,
            "n_misses": n_misses,
            "n_false_alarms": n_false_alarms,
        }
        data_logging_queue.put(
            {"type": "behavioural", "behavioural_data": behavioural_data}
        )

        if running:
            # Display behavioural results.
            show_behavioural_results(
                screen=screen,
                screen_width=screen_width,
                screen_height=screen_height,
                background_color=text_config.rest_condition_color,
                text_color=text_config.font_colors[0],
                n_hits=n_hits,
                n_misses=n_misses,
                n_false_alarms=n_false_alarms,
                display_duration_ms=5000,
            )

        # Send final board data.
        io.drain_eeg()

    except Exception as e:
        print(f"An error occurred during the experiment: {e}")
        print(traceback.format_exc())
    finally:
        pygame.quit()
        print("Experiment closed.")

    eeg_device.stop_stream()
    eeg_device.release_session()

    print("Join process for sending data")
    data_logging_queue.put(None)
    logging_process.join()
