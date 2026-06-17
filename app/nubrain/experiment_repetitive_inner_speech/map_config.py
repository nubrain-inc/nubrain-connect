def map_session_config_ris_condition(  # ris = repetitive inner speech
    *,
    session_config: dict,
    experiment_config: dict,
):
    """
    Map session config to experiment config. Specifically:
    - subject (integer), e.g. 1 -> subject_id (string), e.g. "sub-001"
    - session (integer), e.g. 1 -> session_id (string), e.g. "session-001"
    """
    subject = session_config["subject"]
    session = session_config["session"]

    subject_id = f"sub-{subject:03}"
    session_id = f"session-{session:03}"

    experiment_config["subject_id"] = subject_id
    experiment_config["session_id"] = session_id

    return experiment_config
