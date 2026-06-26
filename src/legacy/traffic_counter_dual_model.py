import os
import csv
import time
import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter

# ---------------- GDDKiA Constants ----------------
GDDKIA_LABELS = ['a_bikes', 'b_moto', 'c_pass', 'd_light_comm', 
                 'e_heavy_rigid', 'f_articulated', 'g_bus', 'h_agri']

# ---------------- Helpers ----------------
def line_side(px, py, ax, ay, bx, by):
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax)

def draw_line(frame, p1, p2, count_A_to_B, count_B_to_A):
    cv2.line(frame, p1, p2, (0, 255, 255), 2)
    label = f"A->B: {count_A_to_B} | B->A: {count_B_to_A}"
    cv2.putText(frame, label, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

# ---------------- Classifier Wrapper ----------------
class GDDKIAClassifier:
    def __init__(self, model_path, device):
        self.device = device
        # self.model = models.convnext_base(weights=None)
        self.model = models.convnext_small(weights=None)
        self.model.classifier[2] = nn.Linear(self.model.classifier[2].in_features, len(GDDKIA_LABELS))
        
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    @torch.inference_mode()
    def predict(self, crop):
        input_tensor = self.transform(crop).unsqueeze(0).to(self.device)
        outputs = self.model(input_tensor)
        _, pred = torch.max(outputs, 1)
        return pred.item()

# ---------------- Event Engine ----------------
class TrafficEngine:
    def __init__(self, p1, p2, classifier, classify_interval=5):
        self.p1, self.p2 = p1, p2
        self.classifier = classifier
        self.classify_interval = max(1, classify_interval)
        self.last_side = {} 
        self.track_votes = defaultdict(Counter)
        self.track_frame_count = defaultdict(int)
        self.counted = set()
        self.stats = {"A->B": 0, "B->A": 0}

    def process_track(self, tid, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        cx, cy = (x1 + x2)/2, (y1 + y2)/2
        
        # Update Classifier Votes
        h, w, _ = frame.shape
        pad = 10
        crop = frame[max(0, y1-pad):min(h, y2+pad), max(0, x1-pad):min(w, x2+pad)]
        
        if crop.size > 0:
            self.track_frame_count[tid] += 1
            # Here, -1 to also predict on the first frame 
            if (self.track_frame_count[tid] - 1) % self.classify_interval == 0:
                pred_label = self.classifier.predict(crop)
                self.track_votes[tid][pred_label] += 1

        # Current best label (available every frame)
        current_label = ""
        if tid in self.track_votes and len(self.track_votes[tid]) > 0:
            most_common_idx = self.track_votes[tid].most_common(1)[0][0]
            current_label = GDDKIA_LABELS[most_common_idx]

        # Check Line Crossing
        side_now = np.sign(line_side(cx, cy, *self.p1, *self.p2))
        prev = self.last_side.get(tid, side_now)
        
        crossed, direction = 0, ""
        
        if prev != 0 and side_now != 0 and np.sign(prev) != np.sign(side_now):
            direction = "A->B" if prev < 0 and side_now > 0 else "B->A"
            if tid not in self.counted:
                self.counted.add(tid)
                self.stats[direction] += 1
                crossed = 1

                # cleanup after successful counting -
                # currently commented out, losing track by tracing algorithm will result in re-counting if the same ID is reused, but it can be useful for long videos with many unique IDs
                
                # self.track_votes.pop(tid, None)
                # self.last_side.pop(tid, None)
                # self.track_frame_count.pop(tid, None)
                

        self.last_side[tid] = side_now

        

        return crossed, direction, current_label

# ---------------- Main ----------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    yolo_model = YOLO(args.model)
    gddkia_clf = GDDKIAClassifier(args.clf_weights, device)
    engine = TrafficEngine((args.line[0], args.line[1]), (args.line[2], args.line[3]), gddkia_clf, getattr(args, "classify_interval", 5))

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_name = os.path.splitext(os.path.basename(args.video))[0]
    run_id = f"{timestamp}_{video_name}"
    
    base_dir = Path(args.out)
    video_dir = base_dir / "video"
    csv_dir = base_dir / "vehicle_summary"
    
    video_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    
    video_out_path = video_dir / f"annotated_{run_id}.mp4"
    csv_out_path = csv_dir / f"summary_{run_id}.csv"
    
    print(f"--- Run ID: {run_id} ---")
    print(f"Video Output: {video_out_path}")
    print(f"CSV Output:   {csv_out_path}")
    
    writer = cv2.VideoWriter(str(video_out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    csv_file = open(csv_out_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame", "vehicle_id", "gddkia_class", "direction"])

    frame_idx = 0

    stream = yolo_model.track(source=args.video, classes=args.classes, tracker=args.tracker, conf=args.conf, iou=args.iou, persist=True, stream=True, verbose=False)

    print(f"Starting processing... (Target: {args.stop_frame if args.stop_frame > 0 else 'End'})")

    for results in stream:
        frame_idx += 1
        if args.stop_frame > 0 and frame_idx > args.stop_frame: break
        
        annotated = results.orig_img.copy()
        boxes = results.boxes
        
        if boxes is not None and boxes.id is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            ids = boxes.id.cpu().numpy().astype(int)

            for box, tid in zip(xyxy, ids):
                crossed, direct, label = engine.process_track(tid, results.orig_img, box)
                
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated, f"ID:{tid}, label:{label}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

                if crossed:
                    print(f"[EVENT] Frame {frame_idx}: Vehicle {tid} ({label}) moved {direct}")
                    csv_writer.writerow([frame_idx, tid, label, direct])
                    cv2.putText(annotated, f"CROSSED: {label}", (x1, y2+20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        draw_line(annotated, (args.line[0], args.line[1]), (args.line[2], args.line[3]), 
                  engine.stats["A->B"], engine.stats["B->A"])
        writer.write(annotated)
        if frame_idx % 100 == 0: print(f"Processed {frame_idx} frames...")

    cap.release()
    writer.release()
    csv_file.close()
    print(f"Processing finished. Results in {args.out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--model", type=str, default="../yolo11s.pt")
    parser.add_argument("--clf_weights", type=str, required=True)
    parser.add_argument("--tracker", type=str, default="../botsort.yaml")
    parser.add_argument("--classes", type=int, nargs="+", default=[1, 2, 3, 5, 7], help="YOLO class IDs to track")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="YOLO IoU threshold")
    parser.add_argument("--line", type=int, nargs=4, default=[200, 200, 1800, 1000])
    parser.add_argument("--out", type=str, default="../runs/traffic")
    parser.add_argument("--stop_frame", type=int, default=0, help="Stop after N frames (0 for full video)")
    parser.add_argument("--classify_interval", type=int, default=5, help="Run classifier every N tracked frames")
    args = parser.parse_args()
    main(args)
