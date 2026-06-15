import ctypes
import os
import sys
import tkinter as tk
from pathlib import Path
from tkinter import font, ttk

import yaml


def enable_dpi_awareness():
    """
    Tell Windows to report true pixel dimensions.

    If Windows desktop scaling is set to e.g. 150%, then SetProcessDpiAwareness(1) will
    report the true 4K pixel count rather than the scaled-down logical size. Without the
    DPI call, a 4K screen at 150% scaling would report itself as ~2560×1440, and the GUI
    would end up smaller than intended.
    """
    if sys.platform == "win32":
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass


class SessionConfigEditor:
    """GUI for editing experiment parameters (loaded from yaml file) before launch."""

    # Only these keys are shown in the GUI (and in this order)
    EDITABLE_FIELDS = [
        "subject",
        "session",
        "run",
    ]

    # Target fraction of screen real estate
    WIDTH_FRACTION = 0.3
    HEIGHT_FRACTION = 0.2

    def __init__(self, *, session_config_path: str):
        self.session_config_path = Path(session_config_path)
        with open(self.session_config_path, "rb") as file:
            self.session_config = yaml.safe_load(file)

        self.start_button_press = False

        enable_dpi_awareness()
        self.root = tk.Tk()
        self.root.title("NuBrain Experiment Configuration")
        self._scale_to_screen()
        self._build_gui()

    def _scale_to_screen(self):
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()

        win_w = int(screen_w * self.WIDTH_FRACTION)
        win_h = int(screen_h * self.HEIGHT_FRACTION)

        # Centre the window on screen.
        x = (screen_w - win_w) // 2
        y = (screen_h - win_h) // 2
        self.root.geometry(f"{win_w}x{win_h}+{x}+{y}")

        # Scale font size relative to screen height. ~14px looks good at 1080p
        # (height=1080), so use that as the anchor.
        base_font_size = max(10, int(screen_h / 1080 * 14))
        self.ui_font = font.Font(family="Segoe UI", size=base_font_size)
        self.heading_font = font.Font(
            family="Segoe UI", size=int(base_font_size * 1.3), weight="bold"
        )

        # Apply to all ttk widgets via a style.
        style = ttk.Style()
        style.configure("TLabel", font=self.ui_font)
        style.configure("TButton", font=self.ui_font)
        style.configure("TEntry", font=self.ui_font)

        # ttk.Entry doesn't always pick up the style font for the typed text, so set the
        # default font for Entry explicitly.
        self.root.option_add("*TEntry*Font", self.ui_font)

        # Store for later use in widget creation
        self.pad_x = max(10, int(screen_w * 0.008))
        self.pad_y = max(5, int(screen_h * 0.005))

    def _build_gui(self):
        self.entries = {}

        # Let column 1 (the entries) expand to fill available width.
        self.root.columnconfigure(1, weight=1)

        for i, key in enumerate(self.EDITABLE_FIELDS):
            ttk.Label(self.root, text=key).grid(
                row=i, column=0, padx=self.pad_x, pady=self.pad_y, sticky="e"
            )
            entry = ttk.Entry(self.root)
            entry.insert(0, str(self.session_config.get(key, "")))
            entry.grid(row=i, column=1, padx=self.pad_x, pady=self.pad_y, sticky="ew")
            self.entries[key] = entry

        btn_frame = ttk.Frame(self.root)
        btn_frame.grid(
            row=len(self.EDITABLE_FIELDS), column=0, columnspan=2, pady=self.pad_y * 3
        )
        ttk.Button(btn_frame, text="Start Experiment", command=self._on_start).pack(
            side="left", padx=self.pad_x
        )
        ttk.Button(btn_frame, text="Cancel", command=self.root.destroy).pack(
            side="left", padx=self.pad_x
        )

    def _on_start(self):
        self.start_button_press = True
        # Write GUI values back into the config dict.
        for key, entry in self.entries.items():
            value = entry.get()
            self.session_config[key] = value

        self.root.destroy()

    def input_value_validation(self):
        """All values (subject, session, run) have to be positive integers."""
        validated_session_config = {}
        for key, value in self.session_config.items():
            # Even if the user enters integer values, they might be represented as
            # strings at this point.
            try:
                value = int(value)
            except ValueError:
                pass  # Keep original value for error message
            if not isinstance(value, int):
                error_msg = (
                    f"Session config parameter '{key}' has to be an integer, "
                    f"got {type(value)}"
                )
                raise ValueError(error_msg)
            if value < 1:
                error_msg = (
                    f"Session config parameter '{key}' has to be an integer greater "
                    f"than zero, got {value}"
                )
                raise ValueError(error_msg)
            validated_session_config[key] = value
        self.session_config = validated_session_config

    def run(self) -> dict | None:
        """Show the GUI. Returns the edited config dict, or None if cancelled."""
        # Tkinter and pygame (in `main.py`) both want to own the main loop, so we have
        # to close the tkinter window (`self.root.destroy()` and letting `mainloop()`
        # return) before calling `pygame.init()` (in `main.py`). The below `mainloop()`
        # blocks until the window is destroyed, and only then does control flow to
        # `main.py`.
        self.root.mainloop()

        if self.start_button_press:
            # Validate user input.
            self.input_value_validation()

            # Write changes made in GUI back to file.
            with open(self.session_config_path, "w", encoding="utf-8") as file:
                yaml.safe_dump(self.session_config, file)

            return self.session_config

        else:
            # User pressed "Cancel".
            return None


if __name__ == "__main__":
    # DEBUGGING
    session_config_path = os.path.join(os.path.dirname(__file__), "session_config.yaml")

    # Show GUI
    gui = SessionConfigEditor(session_config_path=session_config_path)
    session_config = gui.run()
    if session_config is None:
        print("Cancelled.")
    else:
        print(f"Edited session_config: {session_config}")

    # Map from session config to experiment config.
    if session_config is not None:
        from map_config import map_session_config_to_experiment_config

        experiment_config = {"n_sections_to_show": 5}
        updated_experiment_config = map_session_config_to_experiment_config(
            session_config=session_config,
            experiment_config=experiment_config,
        )
        print(f"updated_experiment_config: {updated_experiment_config}")
