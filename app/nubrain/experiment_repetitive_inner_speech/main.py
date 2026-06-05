import multiprocessing as mp
import os
import random
import traceback
from time import sleep, time

import numpy as np
import pygame

from nubrain.device.device_interface import create_eeg_device
from nubrain.experiment_image.data import eeg_data_logging
from nubrain.experiment_image.image_config import ImageConfig
from nubrain.experiment_image.randomize_conditions import (
    create_balanced_list,
    sample_next_image,
    shuffle_with_repetitions,
)
from nubrain.image.tools import get_all_images, load_and_scale_image
from nubrain.misc.datetime import get_formatted_current_datetime

mp.set_start_method("spawn", force=True)  # Necessary on if running on windows?


def experiment_image(config: dict):
    # ----------------------------------------------------------------------------------
    # *** Get config

    device_type = config["device_type"]
    lsl_stream_name = config.get("lsl_stream_name", "DSI-24")

    subject_id = config["subject_id"]
    session_id = config["session_id"]

    output_directory = config["output_directory"]
    image_directory = config["image_directory"]

    eeg_channel_mapping = config.get("eeg_channel_mapping", None)

    utility_frequency = config["utility_frequency"]

    initial_rest_duration = config["initial_rest_duration"]
    image_duration = config["image_duration"]
    isi_duration = config["isi_duration"]
    isi_jitter = config["isi_jitter"]
    inter_block_grey_duration = config["inter_block_grey_duration"]

    n_blocks = config["n_blocks"]
    images_per_block = config["images_per_block"]
    n_target_events = config["n_target_events"]

    response_window_duration = config["response_window_duration"]

    eeg_device_address = config.get("eeg_device_address", None)

    image_config = ImageConfig()

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

    images_and_categories = get_all_images(image_directory=image_directory)

    if not images_and_categories:
        raise AssertionError(f"Found no images at {image_directory}")
    print(f"Found {len(images_and_categories)} images")

    # ----------------------------------------------------------------------------------
    # *** Create pseudo-random condition order

    # List with all unique image categories (e.g. `["apple", "banana", ...]`).
    image_categories = list(set([x["image_category"] for x in images_and_categories]))

    n_trials = n_blocks * images_per_block

    # Order of image categories.
    trial_order = create_balanced_list(
        image_categories=image_categories,
        target_length=n_trials,
    )
    random.shuffle(trial_order)

    trial_order = shuffle_with_repetitions(
        list_with_duplicates=trial_order,
        repetitions=n_target_events,
        minimize_runs=True,
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
        "image_directory": image_directory,
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
        "image_duration": image_duration,
        "isi_duration": isi_duration,
        "inter_block_grey_duration": inter_block_grey_duration,
        # Experiment structure
        "n_blocks": n_blocks,
        "images_per_block": images_per_block,
        # Misc
        "utility_frequency": utility_frequency,
        "path_out_data": path_out_data,
        "data_logging_queue": data_logging_queue,
    }

    logging_process = mp.Process(target=eeg_data_logging, args=(subprocess_params,))
    logging_process.daemon = True
    logging_process.start()

    # ----------------------------------------------------------------------------------
    # *** Start experiment

    # Performance counters.
    n_hits = 0
    n_false_alarms = 0
    n_total_targets = 0

    running = True
    while running:
        pygame.init()

        # Get screen dimensions and set up full screen.
        screen_info = pygame.display.Info()
        screen_width = screen_info.current_w
        screen_height = screen_info.current_h
        screen = pygame.display.set_mode(
            (screen_width, screen_height), pygame.FULLSCREEN
        )
        pygame.display.set_caption("Image Presentation Experiment")
        pygame.mouse.set_visible(False)

        idx_trial = 0

        try:
            # Initial grey screen.
            pygame.time.wait(100)
            screen.fill(image_config.rest_condition_color)
            pygame.display.flip()
            pygame.time.wait(100)
            screen.fill(image_config.rest_condition_color)
            pygame.display.flip()

            # Clear board buffer.
            _, _ = eeg_device.get_board_data()

            # Pause for specified number of milliseconds.
            pygame.time.delay(int(round(initial_rest_duration * 1000.0)))

            # Block loop.
            for idx_block in range(n_blocks):
                # Image loop (within a block).
                for image_count in range(images_per_block):
                    if not running:  # Check for quit event
                        break

                    # Sample the next image.
                    next_image_category = trial_order[idx_trial]
                    next_image_file_path = sample_next_image(
                        next_image_category=next_image_category,
                        category_to_filepath=category_to_filepath,
                        previous_image_file_path=previous_image_file_path,
                    )

                    # Load the next image.
                    image_and_metadata = None
                    while image_and_metadata is None:
                        image_and_metadata = load_and_scale_image(
                            image_file_path=next_image_file_path,
                            screen_width=screen_width,
                            screen_height=screen_height,
                        )

                    current_image = image_and_metadata["image"]

                    img_rect = current_image.get_rect(
                        center=(screen_width // 2, screen_height // 2)
                    )
                    screen.fill(image_config.rest_condition_color)
                    screen.blit(current_image, img_rect)
                    pygame.display.flip()
                    t_stim_start = time()  # Start of stimulus presentation.

                    # Insert stimulus start marker and get its timestamp.
                    marker_val, marker_ts = eeg_device.insert_marker(
                        image_config.stim_start_marker
                    )
                    if marker_val is not None:
                        data_logging_queue.put(
                            {
                                "type": "marker",
                                "marker_value": marker_val,
                                "timestamp": marker_ts,
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

                    # Determine if the current trial is a target event.
                    is_target_event = False
                    if idx_trial > 0 and next_image_category == previous_image_category:
                        is_target_event = True
                        n_total_targets += 1

                    response_made = False
                    response_time = np.nan
                    response_deadline = t_stim_start + response_window_duration

                    # Wait for image duration, but check for responses continuously.
                    t_stim_end_expected = t_stim_start + image_duration
                    while time() < t_stim_end_expected:
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                running = False
                            if event.type == pygame.KEYDOWN:
                                keydown_time = time()
                                if event.key == pygame.K_ESCAPE:
                                    running = False
                                # Check for space bar press within the response window.
                                if event.key == pygame.K_SPACE and not response_made:
                                    if keydown_time < response_deadline:
                                        response_made = True
                                        response_time = keydown_time - t_stim_start
                                        print(
                                            f"Response time: {round(response_time, 3)}"
                                        )
                                        if is_target_event:
                                            # Hit.
                                            n_hits += 1
                                        else:
                                            # False alarm.
                                            n_false_alarms += 1
                        if not running:
                            break
                    if not running:
                        break

                    # End of stimulus presentation. Display ISI grey screen.
                    screen.fill(image_config.rest_condition_color)
                    pygame.display.flip()
                    t_stim_end_actual = time()

                    marker_val, marker_ts = eeg_device.insert_marker(
                        image_config.stim_end_marker
                    )
                    if marker_val is not None:
                        data_logging_queue.put(
                            {
                                "type": "marker",
                                "marker_value": marker_val,
                                "timestamp": marker_ts,
                            }
                        )

                    eeg_data, eeg_ts = eeg_device.get_board_data()
                    if eeg_data.size > 0:
                        data_logging_queue.put(
                            {
                                "type": "eeg",
                                "eeg_data": eeg_data,
                                "eeg_timestamps": eeg_ts,
                            }
                        )

                    # Time until when to show grey screen (ISI).
                    t_isi_end = (
                        t_stim_end_actual
                        + isi_duration
                        + np.random.uniform(low=0.0, high=isi_jitter)
                    )

                    # Continue checking for late responses or quit events.
                    while time() < t_isi_end:
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                running = False
                            if event.type == pygame.KEYDOWN:
                                keydown_time = time()
                                if event.key == pygame.K_ESCAPE:
                                    running = False
                                # Still check for spacebar presses that are within the
                                # response window for target events.
                                if event.key == pygame.K_SPACE and not response_made:
                                    if keydown_time < response_deadline:
                                        response_made = True
                                        response_time = keydown_time - t_stim_start
                                        print(
                                            f"Response time: {round(response_time, 3)}"
                                        )
                                        if is_target_event:
                                            # Hit.
                                            n_hits += 1
                                        else:
                                            # False alarm.
                                            n_false_alarms += 1
                        if not running:
                            break
                    if not running:
                        break

                    stimulus_data = {
                        "stimulus_start_time": t_stim_start,
                        "stimulus_end_time": t_stim_end_actual,
                        "stimulus_duration_s": t_stim_end_actual - t_stim_start,
                        "image_file_path": next_image_file_path,
                        "image_category": next_image_category,
                        "is_target_event": is_target_event,
                        "response_time_s": response_time,
                    }
                    data_logging_queue.put(
                        {"type": "stimulus", "stimulus_data": stimulus_data}
                    )

                    # Update tracking variables for the next loop iteration.
                    previous_image_file_path = next_image_file_path
                    previous_image_category = next_image_category
                    idx_trial += 1

                if not running:
                    break

                # Send post-stimulus EEG data (to avoid buffer overflow).
                eeg_data, eeg_ts = eeg_device.get_board_data()
                if eeg_data.size > 0:
                    data_logging_queue.put(
                        {"type": "eeg", "eeg_data": eeg_data, "eeg_timestamps": eeg_ts}
                    )

                # Inter-block grey screen.
                screen.fill(image_config.rest_condition_color)
                pygame.display.flip()
                # We already waited for the ISI duration, therefore subtract it from the
                # inter block duration. Avoid negative value in case ISI duration is
                # longer than inter block duration.
                remaining_wait = max((inter_block_grey_duration - isi_duration), 0.0)
                pygame.time.delay(int(round(remaining_wait * 1000.0)))

            # Calculate behavioural results.
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
                screen.fill(image_config.rest_condition_color)

                # Behavioural results title.
                title_font = pygame.font.Font(None, 72)
                title_text = title_font.render("Experiment Complete", True, (0, 0, 0))
                title_rect = title_text.get_rect(
                    center=(screen_width // 2, screen_height // 2 - 150)
                )
                screen.blit(title_text, title_rect)

                # Behavioural results text.
                results_font = pygame.font.Font(None, 56)
                hits_text = results_font.render(f"Hits: {n_hits}", True, (0, 0, 0))
                misses_text = results_font.render(
                    f"Misses: {n_misses}", True, (0, 0, 0)
                )
                false_alarms_text = results_font.render(
                    f"False Alarms: {n_false_alarms}", True, (0, 0, 0)
                )

                # Position and display results
                hits_rect = hits_text.get_rect(
                    center=(screen_width // 2, screen_height // 2 - 20)
                )
                misses_rect = misses_text.get_rect(
                    center=(screen_width // 2, screen_height // 2 + 40)
                )
                false_alarms_rect = false_alarms_text.get_rect(
                    center=(screen_width // 2, screen_height // 2 + 100)
                )

                screen.blit(hits_text, hits_rect)
                screen.blit(misses_text, misses_rect)
                screen.blit(false_alarms_text, false_alarms_rect)

                pygame.display.flip()
                pygame.time.wait(5000)  # Show results for 5 seconds

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
