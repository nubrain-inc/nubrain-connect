import argparse
import os

from nubrain.experiment_image.load_config import load_config_image_yaml
from nubrain.experiment_repetitive_inner_speech.data import (
    load_config_repetitive_inner_speech_yaml,
)
from nubrain.experiment_text_comprehension.demo import text_demo_comprehension
from nubrain.experiment_text_comprehension.load_experiment_config import (
    load_config_text_comprehension_yaml,
)
from nubrain.experiment_text_comprehension.map_config import (
    map_session_config_comprehension_condition,
)
from nubrain.experiment_text_targets.demo import text_demo_targets
from nubrain.experiment_text_targets.gui import SessionConfigEditor
from nubrain.experiment_text_targets.load_experiment_config import (
    load_config_text_targets_yaml,
)
from nubrain.experiment_text_targets.map_config import (
    map_session_config_target_condition,
)

# Wrap these imports in try, so that the other modules can be imported without
# dependency on pylsl for demo mode.
try:
    from nubrain.experiment_eeg_to_image_v1.load_config import (
        load_config_yaml_eeg_to_image_v1,
    )
    from nubrain.experiment_eeg_to_image_v1.main import experiment_eeg_to_image_v1
    from nubrain.experiment_eeg_to_image_v1.main_autoregressive import (
        experiment_eeg_to_image_v1_autoregressive,
    )
    from nubrain.experiment_image.main import experiment_image
    from nubrain.experiment_repetitive_inner_speech.main import (
        experiment_repetitive_inner_speech,
    )
    from nubrain.experiment_text_comprehension.main import experiment_text_comprehension
    from nubrain.experiment_text_targets.main import experiment_text_targets
except Exception as e:
    load_config_yaml_eeg_to_image_v1 = None
    experiment_eeg_to_image_v1 = None
    experiment_eeg_to_image_v1_autoregressive = None
    experiment_image = None
    experiment_repetitive_inner_speech = None
    experiment_text_comprehension = None
    experiment_text_targets = None
    print(f"Failed to import nubrain main module: {e}")
from nubrain.live_demo.main import run_live_demo


def main():
    """
    Main entry point for the nubrain command-line application.
    """
    # Initialize the parser.
    parser = argparse.ArgumentParser(description="nubrain command-line interface.")

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the experiment configuration YAML file.",
    )

    # Which experimental mode to use. Options:
    # - "data_collection_image": Data collection mode for image stimuli.
    # - "demo_text_targets":  Demo mode (without EEG device) with text stimuli. Word
    #    repetitions as target events for attention task.
    # - "data_collection_text_targets":  Data collection mode for text stimuli with
    #   word repetitions as target events for attention task.
    # - "demo_text_comprehension":  Demo mode (without EEG device) with text stimuli.
    #   Comprehension questions at the end of the run.
    # - "data_collection_text_comprehension":  Data collection mode for text stimuli
    #   with comprehension questions at the end of the run.
    # - "demo_repetitive_inner_speech": Data collection mode for repetitive
    #   inner speech paradigm (with text and image stimuli).
    # - "data_collection_repetitive_inner_speech": Data collection mode for repetitive
    #   inner speech paradigm (with text and image stimuli).
    # - "eeg_to_image":  After presenting each image, directly reconstruct the image,
    #   then show the next image.
    # - "eeg_to_image_autoregressive": After presenting an image, directly reconstruct
    #   the image, and then show the reconstructed image as the next stimulus.
    # - "eeg_to_image_live_demo": Use cache.
    parser.add_argument(
        "--mode",
        type=str,
        default="data_collection_text_targets",
        help="Which experimental mode to use",
    )

    args = parser.parse_args()

    print("nubrain")
    print(f"Configuration file provided: {args.config}")

    input_file_path = args.config

    mode = args.mode

    # Load EEG experiment config from yaml file.
    if mode == "data_collection_image":
        # Data collection mode, image stimuli.
        config = load_config_image_yaml(yaml_file_path=input_file_path)
    elif mode in ["demo_text_targets", "data_collection_text_targets"]:
        # Data collection mode, text stimuli, repeat words target events.
        config = load_config_text_targets_yaml(yaml_file_path=input_file_path)
    elif mode in ["demo_text_comprehension", "data_collection_text_comprehension"]:
        # Text stimuli, comprehension questions.
        config = load_config_text_comprehension_yaml(yaml_file_path=input_file_path)
    elif mode in [
        "data_collection_repetitive_inner_speech",
        "demo_repetitive_inner_speech",
    ]:
        # Repetitive inner speech.
        config = load_config_repetitive_inner_speech_yaml(
            yaml_file_path=input_file_path
        )
    elif mode in ["eeg_to_image", "eeg_to_image_autoregressive"]:
        # Live EEG to image generation mode. Use corresponding config file loading
        # function (different parameters than regular data collection).
        config = load_config_yaml_eeg_to_image_v1(yaml_file_path=input_file_path)
    elif mode == "eeg_to_image_live_demo":
        pass
    else:
        raise AssertionError(f"Unknown experimental mode: {mode}")

    # Run experiment.
    if mode == "data_collection_image":
        experiment_image(config=config)
    elif mode in ["demo_text_targets", "data_collection_text_targets"]:
        # Show GUI for user to update session parameters (subject ID, next run).
        session_config_path = os.path.join(
            os.path.dirname(__file__),
            "experiment_text_targets",
            "session_config.yaml",
        )
        gui = SessionConfigEditor(session_config_path=session_config_path)
        session_config = gui.run()
        if session_config is None:
            # User pressed the cancel button.
            print("Cancelled.")
            return None
        # Map values (subject, session, run) from session config to experiment config.
        config = map_session_config_target_condition(
            session_config=session_config,
            experiment_config=config,
        )
        if mode == "data_collection_text_targets":
            experiment_text_targets(config=config)
        elif mode == "demo_text_targets":
            text_demo_targets(config=config)
        else:
            raise AssertionError
    elif mode in ["demo_text_comprehension", "data_collection_text_comprehension"]:
        # Show GUI for user to update session parameters (subject ID, next run).
        session_config_path = os.path.join(
            os.path.dirname(__file__),
            "experiment_text_comprehension",
            "session_config.yaml",
        )
        gui = SessionConfigEditor(session_config_path=session_config_path)
        session_config = gui.run()
        if session_config is None:
            # User pressed the cancel button.
            print("Cancelled.")
            return None
        # Map values (subject, session, run) from session config to experiment config.
        config = map_session_config_comprehension_condition(
            session_config=session_config,
            experiment_config=config,
        )
        if mode == "data_collection_text_comprehension":
            experiment_text_comprehension(config=config)
        elif mode == "demo_text_comprehension":
            text_demo_comprehension(config=config)
        else:
            raise AssertionError
    elif mode in [
        "data_collection_repetitive_inner_speech",
        "demo_repetitive_inner_speech",
    ]:
        # Show GUI for user to update session parameters (subject ID, next run).
        session_config_path = os.path.join(
            os.path.dirname(__file__),
            "experiment_repetitive_inner_speech",
            "session_config.yaml",
        )
        gui = SessionConfigEditor(session_config_path=session_config_path)
        session_config = gui.run()
        if session_config is None:
            # User pressed the cancel button.
            print("Cancelled.")
            return None
        # Map values (subject, session, run) from session config to experiment config.
        config = map_session_config_comprehension_condition(
            session_config=session_config,
            experiment_config=config,
        )
        if mode == "data_collection_repetitive_inner_speech":
            experiment_repetitive_inner_speech(config=config)
        elif mode == "demo_repetitive_inner_speech":
            # demo_repetitive_inner_speech(config=config)
            raise NotImplementedError
        else:
            raise AssertionError
    elif mode == "eeg_to_image":
        experiment_eeg_to_image_v1(config=config)
    elif mode == "eeg_to_image_autoregressive":
        experiment_eeg_to_image_v1_autoregressive(config=config)
    elif mode == "eeg_to_image_live_demo":
        run_live_demo(cache=input_file_path)  # Pickle file path

    return None


if __name__ == "__main__":
    main()
