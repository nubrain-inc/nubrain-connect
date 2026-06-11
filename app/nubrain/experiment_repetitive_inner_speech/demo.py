"""
Demo mode, does not use EEG device. Repetitive inner speech.
"""

import traceback

import numpy as np
import pygame

from nubrain.audio.tone import generate_tone
from nubrain.experiment_common.dummy_io import DummyExperimentIO
from nubrain.experiment_image.randomize_conditions import (
    create_balanced_list,
    sample_next_image,
    sample_with_min_distance,
    shuffle_dicts_with_repetitions,
)
from nubrain.experiment_repetitive_inner_speech.attention_task import run_attention_task
from nubrain.image.tools import get_all_images, load_and_scale_image
from nubrain.text.rendering import render_spaced_text

# import multiprocessing as mp
# mp.set_start_method("spawn", force=True)  # Necessary on if running on windows?


def demo_repetitive_inner_speech(config: dict):
    # ----------------------------------------------------------------------------------
    # *** Get config

    path_stimuli = config["path_stimuli"]

    stimulus_duration = config["stimulus_duration"]
    post_stim_interval = config["post_stim_interval"]
    n_repetitions_per_trial = config["n_repetitions_per_trial"]
    repeat_duration = config["repeat_duration"]
    inter_trial_interval = config["inter_trial_interval"]
    inter_trial_jitter = config["inter_trial_jitter"]
    n_trials_per_block = config["n_trials_per_block"]
    inter_block_rest_duration = config["inter_block_rest_duration"]
    n_blocks = config["n_blocks"]
    n_target_events = config["n_target_events"]

    stimulus_font_name = config["stimulus_font_name"]
    stimulus_font_is_bold = config["stimulus_font_is_bold"]
    stimulus_font_is_italic = config["stimulus_font_is_italic"]
    stimulus_font_size = config["stimulus_font_size"]
    stimulus_font_spacing = config["stimulus_font_spacing"]
    stimulus_font_color = config["stimulus_font_color"]
    background_color = config["background_color"]

    audio_cue_frequency = config["audio_cue_frequency"]
    audio_cue_duration = config["audio_cue_duration"]
    audio_cue_amplitude = config["audio_cue_amplitude"]

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
    object_classes = list(set([x["image_category"] for x in images_and_categories]))

    # We need at least twice as many trials as categories (for one text and one image
    # trial per category).
    n_stimulus_categories = len(object_classes)

    if n_trials < (n_stimulus_categories * 2):
        print(
            f"Adjusting number of trials from {n_trials} to {n_stimulus_categories * 2} "
            "(required for having one text & image trial each per category)"
        )
        n_trials = n_stimulus_categories * 2

    stimuli = []
    for stimulus_class in object_classes:
        for stimulus_type in ["text", "image"]:
            stimuli.append(
                {
                    "stimulus_class": stimulus_class,
                    "stimulus_type": stimulus_type,
                }
            )

    # Order of image categories.
    trial_order = create_balanced_list(
        image_categories=stimuli,
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

    # ----------------------------------------------------------------------------------
    # *** Create target events

    # Indices of target events.
    target_trial_idcs = sample_with_min_distance(
        n_samples=n_target_events,
        lower=5,  # No targets at very beginning
        upper=(n_trials - 5),  # No targets at the very end
        min_distance=1,
    )

    # ----------------------------------------------------------------------------------
    # *** Dummy I/O helper (not using EEG device for demo mode)

    io = DummyExperimentIO()

    # ----------------------------------------------------------------------------------
    # *** Start experiment

    # Count correct and incorrect answers for attention task.
    n_target_events_correct = 0
    n_target_events_incorrect = 0

    running = True

    pygame.init()

    # ----------------------------------------------------------------------------------
    # *** Prepare audio cue for repetitive inner speech

    # Audio mixer buffer size in frames. Smaller -> lower output latency, but too small
    # risks buffer underruns (audible clicks/dropouts). 256 frames @ 44.1 kHz ~= 5.8 ms
    # (was 512 ~= 11.6 ms). If you hear any glitching, raise this (e.g. to 512).
    audio_mixer_buffer = 256
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=audio_mixer_buffer)

    # Get the sample rate from the mixer settings.
    sample_rate = pygame.mixer.get_init()[0]

    # Warm up the audio path so the very first real cue does not pay a one-off
    # channel-allocation / device-wake cost that would inflate trial-1 latency.
    pygame.mixer.set_num_channels(8)
    try:
        _warmup_sound = pygame.sndarray.make_sound(
            np.zeros((int(sample_rate * 0.05), 2), dtype=np.int16)
        )
        _warmup_sound.set_volume(0.0)
        _warmup_sound.play()
    except Exception:
        pass  # Warm-up is best-effort; never let it abort the experiment.

    audio_cue_tone_data = generate_tone(
        frequency=audio_cue_frequency,  # Pitch of the tone in Hz
        duration=audio_cue_duration,  # Duration of audio cue
        amplitude=audio_cue_amplitude,  # Volume, from 0.0 to 1.0
        sample_rate=sample_rate,
    )

    # Create a sound object from the numpy array.
    audio_cue_tone = pygame.sndarray.make_sound(audio_cue_tone_data)

    # ----------------------------------------------------------------------------------
    # *** Prepare audio cues for inter block interval

    # Use an audio cue at the beginning and at the end of the inter-block interval, so
    # the participant can close their eyes / rest.

    # How long before the end of the inter-block interval to play the audio cue.
    pure_tone_end_delay = 1.0

    # Play the tone to cue the end of the inter-block interval x seconds before the end
    # of the inter-block interval. Do not use the audio cue if the inter-block interval
    # is too short.
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
    # *** Prepare font

    stimulus_font = pygame.font.SysFont(
        stimulus_font_name,
        stimulus_font_size,
        bold=stimulus_font_is_bold,
        italic=stimulus_font_is_italic,
    )

    # ----------------------------------------------------------------------------------
    # *** Prepare visual stimulus generation

    # Get screen dimensions and set up full screen.
    screen_info = pygame.display.Info()
    screen_width = screen_info.current_w
    screen_height = screen_info.current_h
    screen = pygame.display.set_mode((screen_width, screen_height), pygame.FULLSCREEN)
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

        # Pause for specified number of milliseconds.
        pygame.time.delay(int(round(inter_block_rest_duration * 1000.0)))

        # Loop over trials.
        for idx_trial in range(n_trials):
            if not running:  # Check for quit event
                break

            # --------------------------------------------------------------------------
            # *** (1) Stimulus presentation (image or word)

            # E.g. "apple".
            stimulus_class = trial_order[idx_trial]["stimulus_class"]
            # "text" or "image".
            stimulus_type = trial_order[idx_trial]["stimulus_type"]

            # Show image.
            if stimulus_type == "image":
                # Sample the next image.
                image_file_path = sample_next_image(
                    next_image_category=stimulus_class,
                    category_to_filepath=category_to_filepath,
                    previous_image_file_path=None,
                )

                # Load the next image.
                image_and_metadata = load_and_scale_image(
                    image_file_path=image_file_path,
                    screen_width=screen_width,
                    screen_height=screen_height,
                )
                if image_and_metadata is None:
                    raise AssertionError(f"Failed to load stimulus: {image_file_path}")

                current_image = image_and_metadata["image"]

                img_rect = current_image.get_rect(
                    center=(screen_width // 2, screen_height // 2)
                )
                screen.fill(background_color)
                screen.blit(current_image, img_rect)

            # Show text.
            else:
                image_file_path = None

                # Clear previous stimulus.
                screen.fill(background_color)

                stimulus_text = render_spaced_text(
                    text=stimulus_class,
                    font=stimulus_font,
                    color=stimulus_font_color,
                    spacing=stimulus_font_spacing,
                )

                stimulus_rect = stimulus_text.get_rect(
                    center=(screen_width // 2, screen_height // 2)
                )
                screen.blit(stimulus_text, stimulus_rect)

            pygame.display.flip()
            # Start of stimulus presentation.
            t_stim_start = io.now()

            # Wait for image duration, but check for responses continuously.
            if io.wait_until(t_stim_start + stimulus_duration):
                running = False
                break

            # --------------------------------------------------------------------------
            # *** (2) Post-stimulus delay

            # End of stimulus presentation. Display empty screen.
            screen.fill(background_color)
            pygame.display.flip()
            t_stim_end_actual = io.now()

            # Show empty screen for the post-stimulus delay.
            if io.wait_until(t_stim_end_actual + post_stim_interval):
                running = False
                break

            # --------------------------------------------------------------------------
            # *** (3) Silent speech

            silent_speech_cue_onsets = []

            for idx_repetition in range(n_repetitions_per_trial):
                # Play the tone that cues the beginning of an inner speech repetition.
                # Note that `.play()` is non-blocking, so the timestamp below is the
                # command time, not the acoustic onset.
                audio_cue_tone.play()
                t_cue_start = io.now()

                silent_speech_cue_onsets.append(t_cue_start)

                # Wait out this repetition, checking for quit events.
                if io.wait_until(t_cue_start + repeat_duration):
                    running = False
                    break

            if not running:
                break

            # --------------------------------------------------------------------------
            # *** (4) Attention task

            if idx_trial in target_trial_idcs:
                is_target = True

                result = run_attention_task(
                    io=io,
                    screen=screen,
                    screen_width=screen_width,
                    screen_height=screen_height,
                    background_color=background_color,
                    stimulus_font=stimulus_font,
                    stimulus_font_color=stimulus_font_color,
                    stimulus_class=stimulus_class,
                    stimulus_type=stimulus_type,
                    object_classes=object_classes,
                )

                attention_task_log = result["log"]

                if result["log"]["is_correct"] is True:
                    n_target_events_correct += 1
                elif result["log"]["is_correct"] is False:
                    n_target_events_incorrect += 1

                if result["quit_requested"]:
                    # Quit during the attention task.
                    running = False
                    break
            else:
                # Not a target trial.
                is_target = False
                attention_task_log = None

            # --------------------------------------------------------------------------
            # *** (5) Inter-trial interval

            # Inter-stimulus interval. Display empty screen.
            screen.fill(background_color)
            pygame.display.flip()

            # Time until when to show empty screen (post-stimulus delay).
            t_isi_end = (
                io.now()
                + inter_trial_interval
                + np.random.uniform(low=0.0, high=inter_trial_jitter)
            )

            if io.wait_until(t_isi_end):
                running = False
                break

            # --------------------------------------------------------------------------
            # *** (6) Inter-block interval

            if ((idx_trial + 1) % n_trials_per_block) == 0:
                # Until when to stay in the inter-block interval.
                t_ibi_end = io.now() + inter_block_rest_duration

                # Do not play tone if this is the end of the run.
                if (idx_trial + 1) == n_trials:
                    print("Inter-block interval, end of run, disable IBI audio cue")
                    use_ibi_audio_cue = False

                t_ibi_end_audio_cue = None
                if use_ibi_audio_cue:
                    # Audio cue to signal the beginning of the inter-block interval.
                    pure_tone_start.play()
                    # Time when to play the cue signalling the end of the interval.
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
                        pure_tone_end.play()
                        ibi_state["end_cue_played"] = True

                # Pump the event queue (via io.wait_until), so we can quit during rest.
                if io.wait_until(t_ibi_end, on_tick=ibi_tick):
                    running = False

                if not running:
                    break

    except Exception as e:
        print(f"An error occurred during the experiment: {e}")
        print(traceback.format_exc())
    finally:
        pygame.quit()
        print("Experiment closed.")
