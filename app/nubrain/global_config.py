class GlobalConfig:
    def __init__(self, version: str = "v1"):
        self.config_version = version
        # Markers for stimulus start and end (will be stored in marker channel).
        self.stim_start_marker = 1.0
        self.stim_end_marker = 2.0
        # Markers for cue start and end (e.g. audio cue during repetitive inner speech).
        self.cue_start_marker = 3.0
        self.cue_end_marker = 4.0
        # Data type for EEG data to use when saving to hdf5 file.
        self.hdf5_dtype = "float64"
