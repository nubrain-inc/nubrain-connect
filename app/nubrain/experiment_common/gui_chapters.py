import ctypes
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


class SessionConfigEditorChapters:
    """
    GUI for editing experiment parameters (loaded from yaml file) before launch,
    including dropdown for selecting chapters.
    """

    def __init__(self, *, session_config_path: str, chapters: dict):

        # Only these keys are shown in the GUI (and in this order)
        self.EDITABLE_FIELDS = [
            "subject",
            "session",
            "chapter",
            "run",
        ]

        # Of the editable fields, these must be positive integers.
        self.INTEGER_FIELDS = {"subject", "session", "run"}

        # Target fraction of screen real estate
        self.WIDTH_FRACTION = 0.3
        self.HEIGHT_FRACTION = 0.2

        self.session_config_path = Path(session_config_path)
        with open(self.session_config_path, "rb") as file:
            self.session_config = yaml.safe_load(file)

        # Make this one the first option in the dropdown menu.
        FIRST_CHAPTER_NAME = "THE ADVENTURE OF THE PRIORY SCHOOL"
        chapter_names = sorted([x["chapter_name"] for x in chapters])
        if FIRST_CHAPTER_NAME in chapter_names:
            chapter_names.remove(FIRST_CHAPTER_NAME)
            chapter_names = [FIRST_CHAPTER_NAME] + chapter_names
        self.CHAPTER_NAMES = chapter_names

        self.chapters = chapters
        # `self.chapters` is a list of dicts:
        # [
        #     {
        #         "chapter_name": "Example Name",
        #         "n_runs": 10,
        #         "path_json": "/path/to/file.json",
        #     },
        #     ...,
        # ]
        # Create mapping from chapter names to number of runs available for that
        # chapter:
        # {"Example Name": 10, ...}
        self.runs_per_chapter = {x["chapter_name"]: x["n_runs"] for x in self.chapters}
        # Create mapping from chapter names to JSON input files:
        self.json_paths = {x["chapter_name"]: x["json_paths"] for x in self.chapters}

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
        style.configure("TCombobox", font=self.ui_font)
        # The dropdown list itself is a tk Listbox, themed via the option database.
        self.root.option_add("*TCombobox*Listbox*Font", self.ui_font)

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
            if key == "chapter":
                widget = ttk.Combobox(
                    self.root, values=self.CHAPTER_NAMES, state="readonly"
                )
                current = self.session_config.get(key, "")
                if current in self.CHAPTER_NAMES:
                    widget.set(current)
                else:
                    widget.set(self.CHAPTER_NAMES[0])  # Default to first option.
            else:
                widget = ttk.Entry(self.root)
                widget.insert(0, str(self.session_config.get(key, "")))

            widget.grid(row=i, column=1, padx=self.pad_x, pady=self.pad_y, sticky="ew")
            self.entries[key] = widget

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
        """
        Subject, session, run must be positive integers, chapter must be a valid option.
        """
        validated_session_config = {}
        for key, value in self.session_config.items():
            if key in self.INTEGER_FIELDS:
                # Even if the user enters integer values, they might be represented as
                # strings at this point.
                try:
                    value = int(value)
                except ValueError:
                    pass  # Keep original value for error message
                if not isinstance(value, int):
                    raise ValueError(
                        f"Session config parameter '{key}' has to be an integer, "
                        f"got {type(value)}"
                    )
                if value < 1:
                    raise ValueError(
                        f"Session config parameter '{key}' has to be an integer greater"
                        f" than zero, got {value}"
                    )

            elif key == "chapter":
                if value not in self.CHAPTER_NAMES:
                    raise ValueError(
                        f"Session config parameter 'chapter' must be one of "
                        f"{self.CHAPTER_NAMES}, got {value!r}"
                    )

            if key == "run":
                # Check if the selected chapter has enough text sections for the
                # selected number of runs. Because the run input value 1 corresponds to
                # the text section with index 0, `selected_run` can be less than or
                # equal to `available_runs`.
                selected_run = value
                available_runs = self.runs_per_chapter[self.session_config["chapter"]]
                if available_runs < selected_run:
                    raise ValueError(
                        f"Chapter {self.session_config['chapter']} has up to "
                        f"{available_runs} runs, but {selected_run} was selected. "
                        "Choose a lower value or go to the next chapter."
                    )

            validated_session_config[key] = value

        # Select the JSON file with text sections, questions, and answers corresponding
        # to the chapter the participant has chosen.
        validated_session_config["path_stimuli"] = self.json_paths[
            validated_session_config["chapter"]
        ]

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
