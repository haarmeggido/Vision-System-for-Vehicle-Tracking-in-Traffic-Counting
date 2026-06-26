import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import os
import argparse
from pathlib import Path
import threading

import legacy.traffic_counter_dual_model as processor

class TrafficGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("GDDKiA Traffic Counter Launcher")
        self.root.geometry("600x650")

        # --- Defaults Confis, modify as needed ---
        self.defaults = {
            "video": r".\data\videos\sample_video.mp4",
            "model": "../yolo11s.pt",
            "clf_weights": r".\src\runs\classifier_convnext\best_model.pth",
            "tracker": "../botsort.yaml",
            "classes": "1,2,3,5,7",
            "conf": "0.25",
            "iou": "0.5",
            "out": "../runs/traffic",
            "stop_frame": "0",
            "classify_interval": "5",
            "line": [200, 200, 1800, 1000]
        }
        
        # Helper to store line coords
        self.line_coords = self.defaults["line"]
        self.line_status_var = tk.StringVar(value="Line: Default (200,200) -> (1800,1000)")

        self._create_widgets()

    def _create_widgets(self):
        # Title
        tk.Label(self.root, text="Traffic Counter Configuration", font=("Arial", 16, "bold")).pack(pady=10)

        # Container for inputs
        frame = tk.Frame(self.root)
        frame.pack(padx=20, pady=5, fill="x")

        # --- File Selectors ---
        self.entry_video = self._add_file_selector(frame, "Input Video:", self.defaults["video"], is_file=True)
        self.entry_model = self._add_file_selector(frame, "YOLO Model:", self.defaults["model"], is_file=True)
        self.entry_clf = self._add_file_selector(frame, "Classifier Weights:", self.defaults["clf_weights"], is_file=True)
        self.entry_tracker = self._add_file_selector(frame, "Tracker Config:", self.defaults["tracker"], is_file=True)
        
        # --- Settings ---
        tk.Frame(self.root, height=1, bg="grey").pack(fill="x", padx=20, pady=10)
        
        settings_frame = tk.Frame(self.root)
        settings_frame.pack(padx=20, fill="x")

        # YOLO classes
        tk.Label(settings_frame, text="YOLO Class IDs:").grid(row=0, column=0, sticky="w")
        self.entry_classes = tk.Entry(settings_frame, width=40)
        self.entry_classes.insert(0, self.defaults["classes"])
        self.entry_classes.grid(row=0, column=1, padx=5, pady=5)

        # Confidence threshold
        tk.Label(settings_frame, text="YOLO Confidence:").grid(row=1, column=0, sticky="w")
        self.entry_conf = tk.Entry(settings_frame, width=10)
        self.entry_conf.insert(0, self.defaults["conf"])
        self.entry_conf.grid(row=1, column=1, sticky="w", padx=5, pady=5)

        # IoU threshold
        tk.Label(settings_frame, text="YOLO IoU:").grid(row=2, column=0, sticky="w")
        self.entry_iou = tk.Entry(settings_frame, width=10)
        self.entry_iou.insert(0, self.defaults["iou"])
        self.entry_iou.grid(row=2, column=1, sticky="w", padx=5, pady=5)

        # Output Dir
        tk.Label(settings_frame, text="Output Directory:").grid(row=3, column=0, sticky="w")
        self.entry_out = tk.Entry(settings_frame, width=40)
        self.entry_out.insert(0, self.defaults["out"])
        self.entry_out.grid(row=3, column=1, padx=5, pady=5)
        
        # Stop Frame
        tk.Label(settings_frame, text="Stop Frame (0=All):").grid(row=4, column=0, sticky="w")
        self.entry_stop = tk.Entry(settings_frame, width=10)
        self.entry_stop.insert(0, self.defaults["stop_frame"])
        self.entry_stop.grid(row=4, column=1, sticky="w", padx=5, pady=5)

        # Classifier interval
        tk.Label(settings_frame, text="Classify Interval:").grid(row=5, column=0, sticky="w")
        self.entry_classify_interval = tk.Entry(settings_frame, width=10)
        self.entry_classify_interval.insert(0, self.defaults["classify_interval"])
        self.entry_classify_interval.grid(row=5, column=1, sticky="w", padx=5, pady=5)

        # --- Line Drawer ---
        tk.Frame(self.root, height=1, bg="grey").pack(fill="x", padx=20, pady=10)
        
        line_frame = tk.Frame(self.root)
        line_frame.pack(pady=5)
        
        tk.Button(line_frame, text="Click to Set Counting Line", command=self.open_line_drawer, 
                  bg="#d9d9d9", font=("Arial", 10)).pack(side="left", padx=10)
        tk.Label(line_frame, textvariable=self.line_status_var, fg="blue").pack(side="left")

        # --- Run Button ---
        tk.Button(self.root, text="START PROCESSING", command=self.run_process, 
                  bg="green", fg="white", font=("Arial", 12, "bold"), height=2).pack(side="bottom", fill="x", padx=20, pady=20)

    def _add_file_selector(self, parent, label_text, default_val, is_file=True):
        row = parent.grid_size()[1]
        tk.Label(parent, text=label_text).grid(row=row, column=0, sticky="w", pady=2)
        
        entry = tk.Entry(parent, width=50)
        entry.insert(0, default_val)
        entry.grid(row=row, column=1, padx=5, pady=2)
        
        btn = tk.Button(parent, text="...", width=3, 
                        command=lambda: self._browse(entry, is_file))
        btn.grid(row=row, column=2, padx=5)
        return entry

    def _browse(self, entry_widget, is_file):
        if is_file:
            path = filedialog.askopenfilename()
        else:
            path = filedialog.askdirectory()
        if path:
            entry_widget.delete(0, tk.END)
            entry_widget.insert(0, path)

    def open_line_drawer(self):
        video_path = self.entry_video.get().strip()
        if not os.path.exists(video_path):
            messagebox.showerror("Error", f"Video file not found:\n{video_path}")
            return

        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            messagebox.showerror("Error", "Could not read the first frame of the video.")
            return

        scale = 1.0
        h, w = frame.shape[:2]
        if w > 1600:
            scale = 1600 / w
            frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        
        self.temp_points = []
        window_name = "Draw Line: Click Start then End point. Press SPACE to Save."
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(self.temp_points) < 2:
                    self.temp_points.append((x, y))
                    # Draw visual feedback
                    cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
                    if len(self.temp_points) == 2:
                        cv2.line(frame, self.temp_points[0], self.temp_points[1], (0, 255, 0), 2)
                    cv2.imshow(window_name, frame)

        cv2.imshow(window_name, frame)
        cv2.setMouseCallback(window_name, mouse_callback)
        
        # Wait for key
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 32: # Space
                if len(self.temp_points) == 2:
                    # Scale back to original resolution
                    p1 = [int(self.temp_points[0][0] / scale), int(self.temp_points[0][1] / scale)]
                    p2 = [int(self.temp_points[1][0] / scale), int(self.temp_points[1][1] / scale)]
                    
                    self.line_coords = [p1[0], p1[1], p2[0], p2[1]]
                    self.line_status_var.set(f"Line: {p1} -> {p2}")
                    break
                else:
                    print("Please select 2 points first.")
            elif key == 27: # Esc
                break
        
        cv2.destroyAllWindows()

    def run_process(self):
        # Gather arguments
        args = argparse.Namespace()
        args.video = self.entry_video.get().strip()
        args.model = self.entry_model.get().strip()
        args.clf_weights = self.entry_clf.get().strip()
        args.tracker = self.entry_tracker.get().strip()
        args.out = self.entry_out.get().strip()
        args.line = self.line_coords

        try:
            args.classes = [int(x) for x in self.entry_classes.get().replace(',', ' ').split() if x.strip()]
            if len(args.classes) == 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Error", "YOLO Class IDs must be a comma-separated list of integers.")
            return

        try:
            args.conf = float(self.entry_conf.get())
        except ValueError:
            messagebox.showerror("Error", "YOLO Confidence must be a number.")
            return

        try:
            args.iou = float(self.entry_iou.get())
        except ValueError:
            messagebox.showerror("Error", "YOLO IoU must be a number.")
            return

        try:
            args.stop_frame = int(self.entry_stop.get())
        except ValueError:
            messagebox.showerror("Error", "Stop Frame must be a number.")
            return

        try:
            args.classify_interval = int(self.entry_classify_interval.get())
            if args.classify_interval < 1:
                raise ValueError
        except ValueError:
            messagebox.showerror("Error", "Classify Interval must be a positive integer.")
            return

        if not os.path.exists(args.video):
            messagebox.showerror("Error", "Video Path Invalid")
            return
        if not os.path.exists(args.model):
            messagebox.showerror("Error", "YOLO Model Path Invalid")
            return
        if not os.path.exists(args.clf_weights):
            messagebox.showerror("Error", "Classifier Weights Invalid")
            return
        if not os.path.exists(args.tracker):
            messagebox.showerror("Error", "Tracker Config Invalid")
            return

        self.root.title("Running... Check Console")
        
        # Run in separate thread to keep UI responsive
        def target():
            try:
                print("\n--- Starting Traffic Counter ---")
                processor.main(args)
                messagebox.showinfo("Success", "Processing Complete!")
            except Exception as e:
                messagebox.showerror("Runtime Error", str(e))
                print(e)
            finally:
                self.root.title("GDDKiA Traffic Counter Launcher")

        threading.Thread(target=target, daemon=True).start()

if __name__ == "__main__":
    root = tk.Tk()
    app = TrafficGUI(root)
    root.mainloop()
