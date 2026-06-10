"""
Shared runtime helpers for nubrain EEG experiment scripts.

These helpers encapsulate the plumbing that is identical across experiment paradigms:

* Pumping the pygame event queue (to keep the window responsive and catch quit requests)
* Waiting until a target time on the EEG/LSL clock
* Emitting stimulus markers (handling the device-specific path)
* Draining the EEG board buffer into the data-logging queue.

The recurring state shared by all of these operations is the triple `(eeg_device,
device_type, data_logging_queue)`. Bundling it in a single object keeps the call sites
at the experiment level short and uniform.
"""

import pygame

# Device types whose markers are inserted directly into the board's time series as
# hardware markers. Other devices (DSI-24) instead receive an LSL-timestamped marker
# routed through the data-logging queue.
_HARDWARE_MARKER_DEVICES = ("cyton", "synthetic")


def pump_events(on_event=None) -> bool:
    """
    Drain the pygame event queue and report whether the run should stop.

    Returns True if a quit was requested (the window was closed or escape was pressed).
    Pumping the queue on every frame is also what keeps the OS from flagging the window
    as unresponsive during long waits, so this should be called from every waiting
    phase (such as ISI, inter-block rest).

    `on_event`, if given, is called with every event that is not a quit / escape (e.g. a
    spacebar press on a target event), which lets a caller collect responses during a
    timed wait. The callback receives the raw pygame event.
    """
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            return True
        if on_event is not None:
            on_event(event)
    return False


class ExperimentIO:
    """
    Plumbing shared across experiment scripts: timing, markers, EEG logging.

    Bundles the `(eeg_device, device_type, data_logging_queue)` triple that every helper
    would otherwise need passed in explicitly. Construct one instance after the
    data-logging subprocess has started, then route all device / clock / marker / EEG
    access through it.
    """

    def __init__(self, *, eeg_device, device_type, data_logging_queue):
        self.eeg_device = eeg_device
        self.device_type = device_type
        self.data_logging_queue = data_logging_queue

    # Clock.
    def now(self) -> float:
        """Current time on the EEG/LSL clock, in seconds."""
        return self.eeg_device.lsl_local_clock()

    # Timed, interruptible waits.
    def wait_until(self, t_end: float, on_tick=None, on_event=None) -> bool:
        """
        Busy-wait until `t_end` (EEG clock), pumping events each iteration.

        `on_tick`, if given, is called once per loop iteration; use it for
        time-dependent side effects during the wait (e.g. a tone scheduled for partway
        through an inter-block interval).

        `on_event`, if given, is called for every non-quit pygame event; use it to
        collect responses (e.g. a button press during an attention trial) during the
        wait. It is forwarded to `pump_events`.

        Returns True if a quit was requested while waiting. Callers should treat True as
        "stop the run" (set their `running` flag and break).
        """
        while self.now() < t_end:
            if pump_events(on_event):
                return True
            if on_tick is not None:
                on_tick()
        return False

    # Markers.
    def emit_marker(self, marker_value, timestamp: float) -> None:
        """
        Record a stimulus marker using the device-appropriate channel.

        Hardware-marker devices get the marker inserted directly into the board's time
        series; other devices get an LSL-timestamped marker via the data-logging queue.
        """
        if self.device_type in _HARDWARE_MARKER_DEVICES:
            self.eeg_device.insert_marker(marker_value)
        else:
            self.data_logging_queue.put(
                {
                    "type": "marker",
                    "marker_value": marker_value,
                    "timestamp": timestamp,
                }
            )

    # EEG buffer handling.
    def drain_eeg(self) -> None:
        """
        Pull buffered EEG samples from the board and log them.

        Called to keep the board's buffer from overflowing.
        """
        eeg_data, eeg_ts = self.eeg_device.get_board_data()
        if eeg_data.size > 0:
            self.data_logging_queue.put(
                {
                    "type": "eeg",
                    "eeg_data": eeg_data,
                    "eeg_timestamps": eeg_ts,
                }
            )

    def discard_eeg(self) -> None:
        """
        Drop any buffered samples without logging them.

        Use this to clear the board buffer at the start of a run, before the first
        trial, so that pre-experiment data is not logged.
        """
        self.eeg_device.get_board_data()
