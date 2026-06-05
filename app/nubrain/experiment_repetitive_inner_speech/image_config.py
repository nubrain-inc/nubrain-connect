from nubrain.global_config import GlobalConfig


class ImageConfig:
    def __init__(self, version: str = "v1"):
        global_config = GlobalConfig()

        self.config_version = version
        # Color values for experimental rest condition (e.g. grey).
        self.rest_condition_color = (128, 128, 128)
        # Markers for stimulus start and end (will be stored in marker channel).
        self.stim_start_marker = global_config.stim_start_marker
        self.stim_end_marker = global_config.stim_end_marker
        # Data type for EEG data to use when saving to hdf5 file.
        self.hdf5_dtype = global_config.hdf5_dtype
        # Resize longest image dimension to this size when saving image to hdf5 file.
        self.max_img_storage_dimension = 128
