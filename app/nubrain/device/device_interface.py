"""
EEG device interface abstraction to support multiple EEG systems (OpenBCI Cyton,
Wearable Sensing DSI-24).
"""

import threading
import time
from abc import ABC, abstractmethod
from typing import Dict, Optional

import numpy as np
from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams
from pylsl import (
    IRREGULAR_RATE,
    StreamInfo,
    StreamInlet,
    StreamOutlet,
    local_clock,
    resolve_byprop,
)


class EEGDeviceInterface(ABC):
    """Abstract interface for EEG devices."""

    @abstractmethod
    def prepare_session(self):
        pass

    @abstractmethod
    def start_stream(self):
        pass

    @abstractmethod
    def stop_stream(self):
        pass

    @abstractmethod
    def get_board_data(self) -> tuple[np.ndarray, np.ndarray]:
        pass

    # Only the OpenBCI Cyton device uses a marker channel, not the WS-DSI-24.
    # @abstractmethod
    # def insert_marker(self, marker: float):
    #     pass

    @abstractmethod
    def release_session(self):
        pass

    @abstractmethod
    def get_device_info(self) -> Dict:
        pass


class BrainFlowDevice(EEGDeviceInterface):
    """
    BrainFlow-based device implementation (OpenBCI Cyton & synthetic).

    When using an OpenBCI device, we insert a stimulus marker into the time series data
    on the EEG board. These markers can be used during analysis to identify stimulus
    events.
    """

    def __init__(
        self,
        *,
        device_type: str,  # "cyton" or "synthetic"
        board_id: int,
        params: BrainFlowInputParams,
        eeg_channel_mapping: Dict[int, str],
    ):
        self.device_type = device_type
        self.board_id = board_id
        self.params = params
        self.board = BoardShim(board_id, params)
        self.eeg_channel_mapping = eeg_channel_mapping
        self.board_description = BoardShim.get_board_descr(board_id)

        # Update channel names from config.
        eeg_channel_idxs = sorted(list(eeg_channel_mapping.keys()))
        eeg_channel_names = [eeg_channel_mapping[idx] for idx in eeg_channel_idxs]
        self.board_description["eeg_names"] = ",".join(eeg_channel_names)

        self.eeg_channel_idxs = eeg_channel_idxs
        self.n_eeg_channels = len(self.eeg_channel_idxs)

        # We don't use LSL for the OpenBCI Cyton device, but we define the local clock
        # for compatibility. We use this clock in the main experiment loop for timing.
        self.lsl_local_clock = local_clock

    def prepare_session(self):
        self.board.prepare_session()

    def start_stream(self):
        self.board.start_stream()

    def stop_stream(self):
        self.board.stop_stream()

    def get_board_data(self) -> tuple[np.ndarray, np.ndarray]:
        board_data = self.board.get_board_data()
        # eeg_data = board_data[self.eeg_channels, :]
        timestamp_channel = board_data[self.timestamp_channel, :]
        return board_data, timestamp_channel

    def insert_marker(self, marker: float):
        self.board.insert_marker(marker)

    def release_session(self):
        self.board.release_session()

    def get_device_info(self) -> Dict:
        # Get actual board data to determine total channels.
        test_data = self.board.get_board_data()
        if test_data.size > 0:
            n_channels_total = test_data.shape[0]
        else:
            n_channels_total = (
                len(self.board_description["eeg_channels"]) + 1
            )  # +1 for marker

        return {
            "board_description": self.board_description,
            "sampling_rate": int(self.board_description["sampling_rate"]),
            "eeg_channels": self.board_description["eeg_channels"],
            "marker_channel": self.board_description["marker_channel"],
            "n_channels_total": n_channels_total,
        }


class DSI24Device(EEGDeviceInterface):
    """
    DSI-24 device implementation using LSL.

    When using a DSI-24 device, we use LSL timestamps stored in the hdf5 file for
    identifying stimulus events (i.e. we do not insert stimulus marker into the time
    series data).
    """

    def __init__(
        self,
        *,
        lsl_stream_name: str = "DSI-24",
        eeg_channel_mapping: Optional[Dict[int, str]] = None,
    ):
        self.lsl_stream_name = lsl_stream_name
        self.eeg_channel_mapping = eeg_channel_mapping
        self.inlet = None
        self.stream_info = None
        self.channel_labels = []
        self.sampling_rate = 0
        self.n_channels = 0

        # Data buffer and threading.
        self.data_buffer = []
        self.timestamps_buffer = []
        self.is_streaming = False
        self.pull_thread = None
        self.buffer_lock = threading.Lock()

        # Import LSL functions for use in other methods.
        self.resolve_byprop = resolve_byprop
        self.StreamInlet = StreamInlet
        self.StreamOutlet = StreamOutlet
        self.StreamInfo = StreamInfo
        self.lsl_local_clock = local_clock
        self.IRREGULAR_RATE = IRREGULAR_RATE

    def prepare_session(self):
        """Connect to the DSI-24 stream and create marker outlet."""
        print(f"Looking for LSL stream with name '{self.lsl_stream_name}'...")

        # Try to resolve by name first, then by type.
        streams = self.resolve_byprop("name", self.lsl_stream_name, timeout=5.0)
        if not streams:
            print(f"No stream found with name '{self.lsl_stream_name}'")
            streams = self.resolve_byprop("type", "EEG", timeout=5.0)

        if not streams:
            raise RuntimeError(
                f"Could not find DSI-24 LSL stream. Make sure DSI-Streamer is running "
                f"and streaming with name '{self.lsl_stream_name}'"
            )

        # Use the first found stream.
        self.stream_info = streams[0]
        print(f"Found stream: {self.stream_info.name()} ({self.stream_info.type()})")

        # Create inlet for receiving EEG data.
        self.inlet = self.StreamInlet(self.stream_info, max_buflen=360)

        # Get full stream info including channel labels.
        full_info = self.inlet.info()
        self.sampling_rate = full_info.nominal_srate()
        self.n_channels = full_info.channel_count()

        expected_sampling_rate = 600
        if int(self.sampling_rate) < expected_sampling_rate:
            raise RuntimeError(
                f"Sampling rate is {self.sampling_rate}, "
                f"expected {expected_sampling_rate} | "
                "Make sure to set correct sampling rate."
            )
        if int(self.sampling_rate) > expected_sampling_rate:
            print(
                f"WARNING: Sampling rate is {self.sampling_rate}, "
                f"expected {expected_sampling_rate}"
            )

        # Try to get channel labels from the stream.
        self.channel_labels = full_info.get_channel_labels()

        if self.eeg_channel_mapping is not None:
            print(
                "WARNING: eeg_channel_mapping from config yaml is ignored when using "
                "DSI-24 device. Will get channel mapping from DSI-24 device."
            )

        # `self.channel_labels` is a list of strings, e.g. `["P3", "C3", "F3", "Fz",
        # ...]`. Construct `eeg_channel_mapping` dictionary, e.g. `{0: 'P3', 1: 'C3', 2:
        # 'F3', 3: 'Fz'}`. We assume that the order of channels in `self.channel_labels`
        # corresponds to the rows of the array of EEG data obtained from the device.
        # TODO: Check if that assumption holds.
        self.eeg_channel_mapping = {}
        for idx_channel, channel_label in enumerate(self.channel_labels):
            self.eeg_channel_mapping[idx_channel] = channel_label

    def start_stream(self):
        """Start pulling data from the inlet in a background thread."""
        if not self.inlet:
            raise RuntimeError("Must call prepare_session() before start_stream()")

        self.is_streaming = True
        self.pull_thread = threading.Thread(target=self._pull_data_loop)
        self.pull_thread.daemon = True
        self.pull_thread.start()
        print("Started streaming from DSI-24")

    def stop_stream(self):
        """Stop the background data pulling thread."""
        self.is_streaming = False
        if self.pull_thread:
            self.pull_thread.join(timeout=2.0)
        print("Stopped streaming from DSI-24")

    def _pull_data_loop(self):
        """Background thread that continuously pulls data from the inlet."""
        while self.is_streaming:
            try:
                # Pull chunk of samples (more efficient than single samples).
                chunk, timestamps = self.inlet.pull_chunk(timeout=0.0, max_samples=1024)

                if timestamps:
                    with self.buffer_lock:
                        self.data_buffer.extend(chunk)
                        self.timestamps_buffer.extend(timestamps)

                # Small sleep to prevent CPU spinning.
                time.sleep(0.001)

            except Exception as e:
                print(f"Error in pull_data_loop: {e}")
                if not self.is_streaming:
                    break

    def get_board_data(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get accumulated data from the buffer and clear it. Returns EEG data and
        corresponding timestamps.
        """
        with self.buffer_lock:
            if not self.data_buffer:
                # Return empty data and timestamps arrays.
                return np.array([]).reshape(self.n_channels, 0), np.array([])

            # Convert buffered data to numpy array. Transpose to get channels x samples.
            data_array = np.array(self.data_buffer).T
            timestamps_array = np.array(self.timestamps_buffer)

            # Clear buffers.
            self.data_buffer.clear()
            self.timestamps_buffer.clear()

            return data_array, timestamps_array

    def release_session(self):
        """Clean up resources."""
        if self.inlet:
            self.inlet.close_stream()
            self.inlet = None

        self.stream_info = None

        with self.buffer_lock:
            self.data_buffer.clear()
            self.timestamps_buffer.clear()

        print("Released DSI-24 session")

    def get_device_info(self) -> Dict:
        """Get device information in format compatible with existing code."""
        if not self.inlet:
            raise RuntimeError("Device not initialized. Call prepare_session() first.")

        # Create board description similar to BrainFlow format.
        board_description = {
            "name": f"DSI-24 ({self.stream_info.name()})",
            "sampling_rate": self.sampling_rate,
            "eeg_names": ",".join(self.channel_labels),
        }

        return {
            "board_description": board_description,
            "sampling_rate": int(self.sampling_rate),
            "eeg_channels": None,  # For backward compatibility with Cyton
            "marker_channel": None,  # For backward compatibility with Cyton
            "n_channels_total": self.n_channels,
        }


def create_eeg_device(device_type: str, **kwargs) -> EEGDeviceInterface:
    """
    Factory function to create EEG device instance.

    Args:
        device_type: 'cyton', 'synthetic', or 'dsi24'
        **kwargs: Device-specific parameters

    Returns:
        EEGDeviceInterface instance
    """
    if device_type == "dsi24":
        return DSI24Device(
            lsl_stream_name=kwargs.get("lsl_stream_name", "DSI-24"),
            eeg_channel_mapping=kwargs.get("eeg_channel_mapping", None),
        )
    elif device_type == "synthetic":
        params = BrainFlowInputParams()
        params.serial_port = kwargs["eeg_device_address"]
        return BrainFlowDevice(
            device_type=device_type,
            board_id=BoardIds.SYNTHETIC_BOARD.value,
            params=params,
            eeg_channel_mapping=kwargs["eeg_channel_mapping"],
        )
    elif device_type == "cyton":
        params = BrainFlowInputParams()
        params.serial_port = kwargs["eeg_device_address"]
        return BrainFlowDevice(
            device_type=device_type,
            board_id=BoardIds.CYTON_BOARD.value,
            params=params,
            eeg_channel_mapping=kwargs["eeg_channel_mapping"],
        )
    else:
        raise ValueError(f"Unknown device type: {device_type}")
