# TODO


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
    raise NotImplementedError
