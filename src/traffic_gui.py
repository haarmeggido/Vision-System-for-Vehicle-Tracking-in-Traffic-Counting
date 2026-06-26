#!/usr/bin/env python3
"""Tkinter GUI launcher for the traffic-counting pipeline.

Provides a graphical interface mirroring the CLI in `run_pipeline.py`.
"""

import os
import re
import shlex
import sys
import threading
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2


SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from traffic_counter.config import (
    DEFAULT_YOLO_CLASSES,
    SUPPORTED_CLASSIFIER_ARCHS,
    PipelineConfig,
)

import run_pipeline

class TrafficGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Traffic Counter Launcher")
        self.root.geometry("900x880")
        self.defaults = {
                "video": "data/videos/validation_video_recodrings/stabilized/VID_20260526_193020.mp4",
                "model": "yolo11s.pt",  
                "clf_weights": "runs/final_model_benchmark_09_06_2026/convnext_base/best_model.pth",
                "tracker": "botsort.yaml",
                "classes": ",".join(map(str, DEFAULT_YOLO_CLASSES)),
                "conf": "0.25",
                "iou": "0.5",
                "out": "runs/baseline_1_convnext_base",
                "stop_frame": "0",
                "classify_interval": "5",
                "line": [400, 100, 1800, 800],
                "same_id_cooldown": "60",
                "duplicate_iou": "0.85",
                "duplicate_window": "2",
                "duplicate_center": "30.0",
            }

        self.line_coords = list(self.defaults["line"])[:]
        self.line_status_var = tk.StringVar(value=self._line_status_text())

        self._create_widgets()

    def _line_status_text(self):
        p = self.line_coords
        return f"Line: ({p[0]},{p[1]}) -> ({p[2]},{p[3]})"

    def _create_widgets(self):
        tk.Label(
            self.root,
            text="Traffic Counter Configuration",
            font=("Arial", 16, "bold"),
        ).pack(pady=10)

        outer = tk.Frame(self.root)
        outer.pack(fill="both", expand=True, padx=12, pady=6)

        canvas = tk.Canvas(outer, borderwidth=0, highlightthickness=0)
        vscroll = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vscroll.set)
        vscroll.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        container = tk.Frame(canvas)
        self._container_window = canvas.create_window((0, 0), window=container, anchor="nw")

        def _on_frame_config(event):
            canvas.configure(scrollregion=canvas.bbox("all"))

        container.bind("<Configure>", _on_frame_config)

        def _on_canvas_config(event):
            canvas.itemconfig(self._container_window, width=event.width)

        canvas.bind("<Configure>", _on_canvas_config)

        def _on_mousewheel(event):
            if hasattr(event, "num") and event.num in (4, 5):
                delta = -1 if event.num == 4 else 1
                canvas.yview_scroll(delta, "units")
                return

            try:
                delta = int(-1 * (event.delta / 120))
            except Exception:
                delta = int(-1 * event.delta)
            canvas.yview_scroll(delta, "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        canvas.bind_all("<Button-4>", _on_mousewheel)
        canvas.bind_all("<Button-5>", _on_mousewheel)

        file_frame = tk.LabelFrame(container, text="Files & Models")
        file_frame.pack(fill="x", padx=6, pady=6)

        self.entry_video = self._add_file_selector(
            file_frame,
            "Input Video:",
            self.defaults["video"],
            is_file=True,
        )
        self.entry_model = self._add_file_selector(
            file_frame,
            "YOLO Model:",
            self.defaults["model"],
            is_file=True,
        )
        self.entry_clf = self._add_file_selector(
            file_frame,
            "Classifier Weights:",
            self.defaults["clf_weights"],
            is_file=True,
        )
        self.entry_tracker = self._add_file_selector(
            file_frame,
            "Tracker Config:",
            self.defaults["tracker"],
            is_file=True,
        )

        settings_frame = tk.LabelFrame(
            container,
            text="Detection / Classification Settings",
        )
        settings_frame.pack(fill="x", padx=6, pady=6)

        tk.Label(settings_frame, text="YOLO Class IDs:").grid(
            row=0,
            column=0,
            sticky="w",
            pady=4,
        )
        self.entry_classes = tk.Entry(settings_frame, width=40)
        self.entry_classes.insert(0, self.defaults["classes"])
        self.entry_classes.grid(row=0, column=1, sticky="w", padx=6)
        tk.Button(
            settings_frame,
            text="Default",
            command=self._set_default_classes,
        ).grid(row=0, column=2, padx=6)

        tk.Label(settings_frame, text="YOLO Confidence:").grid(
            row=1,
            column=0,
            sticky="w",
            pady=4,
        )
        self.entry_conf = tk.Entry(settings_frame, width=10)
        self.entry_conf.insert(0, self.defaults["conf"])
        self.entry_conf.grid(row=1, column=1, sticky="w", padx=6)

        tk.Label(settings_frame, text="YOLO IoU:").grid(
            row=2,
            column=0,
            sticky="w",
            pady=4,
        )
        self.entry_iou = tk.Entry(settings_frame, width=10)
        self.entry_iou.insert(0, self.defaults["iou"])
        self.entry_iou.grid(row=2, column=1, sticky="w", padx=6)

        tk.Label(settings_frame, text="Output Directory:").grid(
            row=3,
            column=0,
            sticky="w",
            pady=4,
        )
        self.entry_out = tk.Entry(settings_frame, width=40)
        self.entry_out.insert(0, self.defaults["out"])
        self.entry_out.grid(row=3, column=1, sticky="w", padx=6)
        tk.Button(
            settings_frame,
            text="Browse",
            command=lambda: self._browse(self.entry_out, is_file=False),
        ).grid(row=3, column=2, padx=6)

        tk.Label(settings_frame, text="Stop Frame (0=All):").grid(
            row=4,
            column=0,
            sticky="w",
            pady=4,
        )
        self.entry_stop = tk.Entry(settings_frame, width=10)
        self.entry_stop.insert(0, self.defaults["stop_frame"])
        self.entry_stop.grid(row=4, column=1, sticky="w", padx=6)

        tk.Label(settings_frame, text="Classify Interval:").grid(
            row=5,
            column=0,
            sticky="w",
            pady=4,
        )
        self.entry_classify_interval = tk.Entry(settings_frame, width=10)
        self.entry_classify_interval.insert(0, self.defaults["classify_interval"])
        self.entry_classify_interval.grid(row=5, column=1, sticky="w", padx=6)

        tk.Label(settings_frame, text="Classifier Arch:").grid(
            row=6,
            column=0,
            sticky="w",
            pady=4,
        )
        default_arch = (
            "convnext_base"
            if "convnext_base" in SUPPORTED_CLASSIFIER_ARCHS
            else SUPPORTED_CLASSIFIER_ARCHS[0]
        )
        self.combo_classifier_arch = ttk.Combobox(
            settings_frame,
            values=SUPPORTED_CLASSIFIER_ARCHS,
            state="readonly",
            width=30,
        )
        self.combo_classifier_arch.set(default_arch)
        self.combo_classifier_arch.grid(row=6, column=1, sticky="w", padx=6)

        tk.Label(settings_frame, text="Voting method:").grid(
            row=7,
            column=0,
            sticky="w",
            pady=4,
        )
        self.combo_voting = ttk.Combobox(
            settings_frame,
            values=["majority", "probability", "probability_yolo_prior"],
            state="readonly",
            width=30,
        )
        self.combo_voting.set("probability_yolo_prior")
        self.combo_voting.grid(row=7, column=1, sticky="w", padx=6)

        tk.Label(settings_frame, text="Line reference:").grid(
            row=8,
            column=0,
            sticky="w",
            pady=4,
        )
        self.combo_line_ref = ttk.Combobox(
            settings_frame,
            values=["centroid", "bottom_center"],
            state="readonly",
            width=30,
        )
        self.combo_line_ref.set("centroid")
        self.combo_line_ref.grid(row=8, column=1, sticky="w", padx=6)

        toggles_frame = tk.LabelFrame(container, text="Optional Toggles")
        toggles_frame.pack(fill="x", padx=6, pady=6)

        self.save_video_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            toggles_frame,
            text="Save annotated video",
            variable=self.save_video_var,
        ).grid(row=0, column=0, sticky="w", padx=6, pady=2)

        self.save_obs_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            toggles_frame,
            text="Save observations (CSV)",
            variable=self.save_obs_var,
        ).grid(row=0, column=1, sticky="w", padx=6, pady=2)

        self.same_id_frame = tk.LabelFrame(container, text="Same-ID Recount")
        self.same_id_frame.pack(fill="x", padx=6, pady=6)

        self.allow_same_id_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            self.same_id_frame,
            text="Allow same ID recount",
            variable=self.allow_same_id_var,
            command=self._toggle_same_id,
        ).grid(row=0, column=0, sticky="w", padx=6, pady=4)

        tk.Label(self.same_id_frame, text="Cooldown (frames):").grid(
            row=0,
            column=1,
            sticky="w",
        )
        self.entry_same_id_cooldown = tk.Entry(self.same_id_frame, width=8)
        self.entry_same_id_cooldown.insert(0, self.defaults["same_id_cooldown"])
        self.entry_same_id_cooldown.grid(row=0, column=2, sticky="w", padx=6)

        self.dup_frame = tk.LabelFrame(container, text="Duplicate Event Suppression")
        self.dup_frame.pack(fill="x", padx=6, pady=6)

        self.suppress_dup_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            self.dup_frame,
            text="Suppress duplicate events",
            variable=self.suppress_dup_var,
            command=self._toggle_dup_options,
        ).grid(row=0, column=0, sticky="w", padx=6, pady=4)

        tk.Label(self.dup_frame, text="IoU threshold:").grid(
            row=1,
            column=0,
            sticky="w",
        )
        self.entry_dup_iou = tk.Entry(self.dup_frame, width=10)
        self.entry_dup_iou.insert(0, self.defaults["duplicate_iou"])
        self.entry_dup_iou.grid(row=1, column=1, sticky="w", padx=6)

        tk.Label(self.dup_frame, text="Frame window:").grid(
            row=1,
            column=2,
            sticky="w",
        )
        self.entry_dup_window = tk.Entry(self.dup_frame, width=6)
        self.entry_dup_window.insert(0, self.defaults["duplicate_window"])
        self.entry_dup_window.grid(row=1, column=3, sticky="w", padx=6)

        tk.Label(self.dup_frame, text="Center distance px:").grid(
            row=1,
            column=4,
            sticky="w",
        )
        self.entry_dup_center = tk.Entry(self.dup_frame, width=10)
        self.entry_dup_center.insert(0, self.defaults["duplicate_center"])
        self.entry_dup_center.grid(row=1, column=5, sticky="w", padx=6)

        line_draw_frame = tk.Frame(container)
        line_draw_frame.pack(fill="x", padx=6, pady=6)

        tk.Button(
            line_draw_frame,
            text="Set Counting Line (open video)",
            command=self.open_line_drawer,
        ).pack(side="left", padx=6)

        tk.Button(
            line_draw_frame,
            text="Reset Line",
            command=self._reset_line,
        ).pack(side="left", padx=6)

        tk.Label(
            line_draw_frame,
            textvariable=self.line_status_var,
            fg="blue",
        ).pack(side="left", padx=12)

        preview_frame = tk.LabelFrame(container, text="Generated CLI Command Preview")
        preview_frame.pack(fill="both", padx=6, pady=6)

        self.command_preview = tk.Text(
            preview_frame,
            height=4,
            wrap="word",
            font=("Courier New", 9),
        )
        self.command_preview.pack(fill="both", padx=6, pady=6)
        self.command_preview.insert(
            "1.0",
            "Validate inputs to generate a reproducible command preview.",
        )
        self.command_preview.config(state="disabled")

        run_frame = tk.Frame(container)
        run_frame.pack(fill="x", padx=6, pady=12)

        self.run_btn = tk.Button(
            run_frame,
            text="START PROCESSING",
            command=self.run_process,
            bg="#2e7d32",
            fg="white",
            font=("Arial", 12, "bold"),
        )
        self.run_btn.pack(side="left", padx=6)

        tk.Button(
            run_frame,
            text="Validate Inputs",
            command=self._validate_inputs,
        ).pack(side="left", padx=6)

        tk.Button(
            run_frame,
            text="Preview Command",
            command=self._preview_command,
        ).pack(side="left", padx=6)

        self._toggle_same_id()
        self._toggle_dup_options()

    def _set_default_classes(self):
        self.entry_classes.delete(0, tk.END)
        self.entry_classes.insert(0, ",".join(map(str, DEFAULT_YOLO_CLASSES)))

    def _add_file_selector(self, parent, label_text, default_val, is_file=True):
        row = parent.grid_size()[1]

        tk.Label(parent, text=label_text).grid(
            row=row,
            column=0,
            sticky="w",
            pady=4,
        )

        entry = tk.Entry(parent, width=56)
        entry.insert(0, default_val)
        entry.grid(row=row, column=1, padx=6)

        tk.Button(
            parent,
            text="...",
            width=3,
            command=lambda: self._browse(entry, is_file),
        ).grid(row=row, column=2, padx=6)

        return entry

    def _browse(self, entry_widget, is_file=True):
        if is_file:
            path = filedialog.askopenfilename()
        else:
            path = filedialog.askdirectory()

        if path:
            entry_widget.delete(0, tk.END)
            entry_widget.insert(0, path)

    def _toggle_same_id(self):
        state = "normal" if self.allow_same_id_var.get() else "disabled"
        self.entry_same_id_cooldown.config(state=state)

    def _toggle_dup_options(self):
        state = "normal" if self.suppress_dup_var.get() else "disabled"
        self.entry_dup_iou.config(state=state)
        self.entry_dup_window.config(state=state)
        self.entry_dup_center.config(state=state)

    def _reset_line(self):
        self.line_coords = list(self.defaults["line"])[:]
        self.line_status_var.set(self._line_status_text())

    def open_line_drawer(self):
        video_path = self.entry_video.get().strip()

        if not video_path:
            messagebox.showerror("Error", "Please select a video file first")
            return

        if not os.path.exists(video_path):
            messagebox.showerror("Error", f"Video file not found:\n{video_path}")
            return

        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            messagebox.showerror("Error", "Could not read the first frame of the video.")
            return

        scale = 1.0
        height, width = frame.shape[:2]

        if width > 1400:
            scale = 1400 / width
            frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

        self.temp_points = []
        window_name = (
            "Draw Line: Click Start then End point. "
            "Press SPACE to Save. ESC to cancel"
        )

        disp = frame.copy()

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(self.temp_points) < 2:
                    self.temp_points.append((x, y))

                    if len(self.temp_points) == 1:
                        cv2.circle(disp, self.temp_points[0], 5, (0, 0, 255), -1)
                    elif len(self.temp_points) == 2:
                        cv2.line(
                            disp,
                            self.temp_points[0],
                            self.temp_points[1],
                            (0, 255, 0),
                            2,
                        )

                    cv2.imshow(window_name, disp)

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, disp)
        cv2.setMouseCallback(window_name, mouse_callback)

        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == 32:
                if len(self.temp_points) == 2:
                    p1 = [
                        int(self.temp_points[0][0] / scale),
                        int(self.temp_points[0][1] / scale),
                    ]
                    p2 = [
                        int(self.temp_points[1][0] / scale),
                        int(self.temp_points[1][1] / scale),
                    ]

                    self.line_coords = [p1[0], p1[1], p2[0], p2[1]]
                    self.line_status_var.set(self._line_status_text())
                    break

                messagebox.showinfo("Info", "Please select 2 points first.")

            elif key == 27:
                break

        cv2.destroyAllWindows()

    def _validate_inputs(self):
        try:
            config = self._gather_config()
            self._update_command_preview(config)
            messagebox.showinfo("OK", "Inputs appear valid")
        except Exception as e:
            messagebox.showerror("Invalid", str(e))

    def _preview_command(self):
        try:
            config = self._gather_config()
            self._update_command_preview(config)
        except Exception as e:
            messagebox.showerror("Invalid", str(e))

    def _gather_config(self) -> PipelineConfig:
        video = self.entry_video.get().strip()
        model = self.entry_model.get().strip()
        clf_weights = self.entry_clf.get().strip()
        tracker = self.entry_tracker.get().strip()
        out_dir = self.entry_out.get().strip()

        if not video:
            raise ValueError("Input video is required")

        if not clf_weights:
            raise ValueError("Classifier weights are required")

        if not out_dir:
            raise ValueError("Output directory is required")

        cls_txt = self.entry_classes.get().strip()
        if not cls_txt:
            raise ValueError("YOLO classes are required")

        try:
            raw = re.split(r"[,\s]+", cls_txt)
            classes = [int(x) for x in raw if x]
            if not classes:
                raise ValueError
        except Exception:
            raise ValueError("YOLO Class IDs must be comma/space separated integers")

        try:
            conf = float(self.entry_conf.get())
        except Exception:
            raise ValueError("YOLO Confidence must be a number")

        if not (0.0 <= conf <= 1.0):
            raise ValueError("YOLO Confidence must be between 0 and 1")

        try:
            iou = float(self.entry_iou.get())
        except Exception:
            raise ValueError("YOLO IoU must be a number")

        if not (0.0 <= iou <= 1.0):
            raise ValueError("YOLO IoU must be between 0 and 1")

        try:
            stop_frame = int(self.entry_stop.get())
        except Exception:
            raise ValueError("Stop Frame must be an integer")

        if stop_frame < 0:
            raise ValueError("Stop Frame must be 0 or positive")

        try:
            classify_interval = int(self.entry_classify_interval.get())
        except Exception:
            raise ValueError("Classify Interval must be an integer")

        if classify_interval < 1:
            raise ValueError("Classify Interval must be a positive integer")

        classifier_arch = self.combo_classifier_arch.get()
        if classifier_arch not in SUPPORTED_CLASSIFIER_ARCHS:
            raise ValueError("Invalid classifier architecture selected")

        voting_method = self.combo_voting.get()
        if voting_method not in ["majority", "probability", "probability_yolo_prior"]:
            raise ValueError("Invalid voting method selected")

        line_reference = self.combo_line_ref.get()
        if line_reference not in ["centroid", "bottom_center"]:
            raise ValueError("Invalid line reference selected")

        try:
            same_id_cooldown = int(self.entry_same_id_cooldown.get())
        except Exception:
            raise ValueError("Same ID cooldown must be an integer")

        if same_id_cooldown < 0:
            raise ValueError("Same ID cooldown must be non-negative")

        try:
            dup_iou = float(self.entry_dup_iou.get())
        except Exception:
            raise ValueError("Duplicate event IoU must be a number")

        if not (0.0 <= dup_iou <= 1.0):
            raise ValueError("Duplicate event IoU must be between 0 and 1")

        try:
            dup_window = int(self.entry_dup_window.get())
        except Exception:
            raise ValueError("Duplicate event window must be an integer")

        if dup_window < 0:
            raise ValueError("Duplicate event frame window must be non-negative")

        try:
            dup_center = float(self.entry_dup_center.get())
        except Exception:
            raise ValueError("Duplicate event center distance must be a number")

        if dup_center < 0:
            raise ValueError("Duplicate center distance must be non-negative")

        if not os.path.exists(video):
            raise FileNotFoundError(f"Video not found: {video}")

        if model and not os.path.exists(model):
            raise FileNotFoundError(f"YOLO model not found: {model}")

        if not os.path.exists(clf_weights):
            raise FileNotFoundError(f"Classifier weights not found: {clf_weights}")

        # Ultralytics can resolve names such as "botsort.yaml", so only enforce
        # existence when the user provides an explicit path.
        if tracker:
            looks_like_explicit_path = "/" in tracker or "\\" in tracker
            if looks_like_explicit_path and not os.path.exists(tracker):
                raise FileNotFoundError(f"Tracker config not found: {tracker}")

        config = PipelineConfig(
            video=Path(video),
            yolo_model=Path(model) if model else Path(self.defaults["model"]),
            classifier_weights=Path(clf_weights),
            tracker=tracker if tracker else self.defaults["tracker"],
            yolo_classes=classes,
            conf=conf,
            iou=iou,
            line=tuple(self.line_coords),
            out_dir=Path(out_dir),
            stop_frame=stop_frame,
            classify_interval=classify_interval,
            classifier_arch=classifier_arch,
            save_video=bool(self.save_video_var.get()),
            voting_method=voting_method,
            line_reference=line_reference,
            save_observations=bool(self.save_obs_var.get()),
            allow_same_id_recount=bool(self.allow_same_id_var.get()),
            same_id_recount_cooldown=same_id_cooldown,
            suppress_duplicate_events=bool(self.suppress_dup_var.get()),
            duplicate_event_iou_threshold=dup_iou,
            duplicate_event_frame_window=dup_window,
            duplicate_event_center_distance=dup_center,
        )

        return config

    def _build_cli_command(self, config: PipelineConfig) -> str:
        parts = [
            "python",
            "src/run_pipeline.py",
            "--video",
            str(config.video),
            "--model",
            str(config.yolo_model),
            "--clf_weights",
            str(config.classifier_weights),
            "--classifier_arch",
            config.classifier_arch,
            "--tracker",
            str(config.tracker),
            "--classes",
            *[str(cls) for cls in config.yolo_classes],
            "--conf",
            str(config.conf),
            "--iou",
            str(config.iou),
            "--line",
            *[str(v) for v in config.line],
            "--voting",
            config.voting_method,
            "--classify_interval",
            str(config.classify_interval),
            "--line_reference",
            config.line_reference,
            "--out",
            str(config.out_dir),
        ]

        if config.stop_frame > 0:
            parts.extend(["--stop_frame", str(config.stop_frame)])

        if config.save_video:
            parts.append("--save_video")

        if config.save_observations:
            parts.append("--save_observations")

        if config.allow_same_id_recount:
            parts.append("--allow_same_id_recount")
            parts.extend([
                "--same_id_recount_cooldown",
                str(config.same_id_recount_cooldown),
            ])

        if config.suppress_duplicate_events:
            parts.append("--suppress_duplicate_events")
            parts.extend([
                "--duplicate_event_iou_threshold",
                str(config.duplicate_event_iou_threshold),
                "--duplicate_event_frame_window",
                str(config.duplicate_event_frame_window),
                "--duplicate_event_center_distance",
                str(config.duplicate_event_center_distance),
            ])

        return " ".join(shlex.quote(part) for part in parts)

    def _update_command_preview(self, config: PipelineConfig):
        command = self._build_cli_command(config)

        self.command_preview.config(state="normal")
        self.command_preview.delete("1.0", tk.END)
        self.command_preview.insert("1.0", command)
        self.command_preview.config(state="disabled")

    def run_process(self):
        try:
            config = self._gather_config()
            self._update_command_preview(config)
        except Exception as e:
            messagebox.showerror("Invalid input", str(e))
            return

        self._set_running_state(True)

        def target():
            error_text = None

            try:
                run_pipeline.process_video(config)
            except Exception:
                error_text = traceback.format_exc()

            self.root.after(0, lambda: self._finish_run(error_text))

        threading.Thread(target=target, daemon=True).start()

    def _set_running_state(self, running: bool):
        if running:
            self.run_btn.config(state="disabled")
            self.root.title("Running... check console for progress")
        else:
            self.run_btn.config(state="normal")
            self.root.title("Traffic Counter Launcher")

    def _finish_run(self, error_text: str | None):
        self._set_running_state(False)

        if error_text is None:
            messagebox.showinfo("Done", "Processing finished successfully")
        else:
            messagebox.showerror("Runtime error", error_text)

if __name__ == "__main__":
    root = tk.Tk()
    app = TrafficGUI(root)
    root.mainloop()
