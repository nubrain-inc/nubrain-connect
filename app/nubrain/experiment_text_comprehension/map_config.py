def map_session_config_comprehension_condition(
    *,
    session_config: dict,
    experiment_config: dict,
):
    """
    Map session config to experiment config. Specifically:
    - subject (integer), e.g. 1 -> subject_id (string), e.g. "sub-001"
    - session (integer), e.g. 1 -> session_id (string), e.g. "session-001"
    - run (integer), e.g. 1 -> section_idx_start (intger) = run - 1
    """
    subject = session_config["subject"]
    session = session_config["session"]
    # chapter = session_config["chapter"]  # Not needed
    run = session_config["run"]

    subject_id = f"sub-{subject:03}"
    session_id = f"session-{session:03}"

    # Minus one because we start at section zero on first run.
    section_idx_start = run - 1

    experiment_config["subject_id"] = subject_id
    experiment_config["session_id"] = session_id
    experiment_config["section_idx_start"] = section_idx_start

    # Path to JSON file containing text sections, questions, and answers for the chapter
    # selected by the user in the GUI.
    experiment_config["path_stimuli"] = session_config["path_stimuli"]

    return experiment_config
