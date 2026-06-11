from time import time

import pygame


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


class DummyExperimentIO:
    """
    Dummy version of ExperimentIO for demo mode (no EEG device used).
    """

    def __init__(self):
        pass

    # Clock.
    def now(self) -> float:
        return time()

    # Timed, interruptible waits.
    def wait_until(self, t_end: float, on_tick=None, on_event=None) -> bool:
        while self.now() < t_end:
            if pump_events(on_event):
                return True
            if on_tick is not None:
                on_tick()
        return False

    def emit_marker(self, marker_value, timestamp: float) -> None:
        pass

    def drain_eeg(self) -> None:
        pass

    def discard_eeg(self) -> None:
        pass
