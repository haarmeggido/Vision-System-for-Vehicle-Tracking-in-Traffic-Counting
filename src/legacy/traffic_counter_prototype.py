"""
Old bit of code, remaining for legacy and reference.
"""


# import os
# import csv
# import time
# import argparse
# import numpy as np
# import cv2
# from ultralytics import YOLO
# import pandas as pd


# # ---------------- Helpers ----------------
# def line_side(px, py, ax, ay, bx, by):
#     """Cross product sign: determines which side of the line (ax,ay)-(bx,by) the point (px,py) is on"""
#     return (bx - ax) * (py - ay) - (by - ay) * (px - ax)

# def draw_line(frame, p1, p2, count_A_to_B, count_B_to_A):
#     cv2.line(frame, p1, p2, (0, 255, 255), 2)
#     label = f"A→B: {count_A_to_B} | B→A: {count_B_to_A}"
#     (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
#     cv2.rectangle(frame, (10,10),(20+tw,20+th),(0,0,0),-1)
#     cv2.putText(frame, label, (15, 25+th-12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),2)

# def put_hud(frame, fps, model_name):
#     text = f"{model_name} | {fps:.1f} FPS"
#     cv2.putText(frame, text, (10, frame.shape[0]-10),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255),2)


# # ---------------- Event Engine ----------------
# class LineCrossCounter:
#     def __init__(self, p1, p2):
#         self.p1 = p1
#         self.p2 = p2
#         self.last_side = {}    # vehicle_id -> last side
#         self.counted = set()
#         self.count_A_to_B = 0
#         self.count_B_to_A = 0

#     def check_cross(self, obj):
#         x1, y1, x2, y2 = obj["bbox"]
#         cx, cy = (x1 + x2)/2, (y1 + y2)/2
#         side_now = np.sign(line_side(cx, cy, *self.p1, *self.p2))
#         prev = self.last_side.get(obj["vehicle_id"], side_now)
#         crossed, direction = 0, ""
#         if prev != 0 and side_now != 0 and np.sign(prev) != np.sign(side_now):
#             direction = "A->B" if prev < 0 and side_now > 0 else "B->A"
#             key = (obj["vehicle_id"], direction)
#             if key not in self.counted:
#                 self.counted.add(key)
#                 crossed = 1
#                 if direction == "A->B":
#                     self.count_A_to_B += 1
#                 else:
#                     self.count_B_to_A += 1
#         self.last_side[obj["vehicle_id"]] = side_now
#         return crossed, direction


# # ---------------- CSV Logger ----------------
# class CSVLogger:
#     def __init__(self, path, class_labels):
#         self.file = open(path, "w", newline="", encoding="utf-8")
#         self.writer = csv.writer(self.file)
#         self.writer.writerow(["frame","time_sec","vehicle_id","class","label",
#                               "confidence","x1","y1","x2","y2","cx","cy","crossed","direction"])
#         self.class_labels = class_labels

#     def log(self, frame_idx, fps, obj, crossed=0, direction=""):
#         x1,y1,x2,y2 = obj["bbox"]
#         cid = obj["class"]
#         label = self.class_labels.get(cid, str(cid))
#         conf = obj["conf"]
#         cx, cy = (x1+x2)/2, (y1+y2)/2
#         self.writer.writerow([frame_idx, f"{frame_idx/fps:.3f}", obj["vehicle_id"],
#                               cid,label,float(conf),x1,y1,x2,y2,int(cx),int(cy),crossed,direction])

#     def close(self):
#         self.file.close()

# def extract_vehicle_summary(input_csv: str, output_csv: str):
#     # Read the events CSV
#     df = pd.read_csv(input_csv)
    
#     # Filter for crossed vehicles
#     df = df[df["crossed"] == 1]
    
#     # Extract relevant columns
#     summary = df[["vehicle_id", "class", "direction", "frame"]].copy()

#     # Map class IDs to category names [TEMPORARY, HARDCODED]
#     class_map = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
#     summary["class"] = summary["class"].map(class_map)
    
#     # Rename columns to match desired output
#     summary.rename(columns={
#         "class": "category",
#         "frame": "time"
#     }, inplace=True)
    
#     # Keep only one row per vehicle_id (first detection)
#     summary = summary.drop_duplicates(subset=["vehicle_id"])
    
#     # Save simplified summary to file
#     summary.to_csv(output_csv, index=False)
#     return summary

# # ---------------- Main ----------------
# def main(args):
#     VEHICLE_CLASS_IDS = [1, 2, 3, 5, 7]  # COCO: bike, car, motorcycle, bus, truck
#     CLASS_LABELS = {1: "bike", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

#     # Output paths
#     os.makedirs(args.out, exist_ok=True)
#     idx = len(os.listdir(args.out)) // 2
#     video_out = os.path.join(args.out, f"annotated_{idx}.mp4")
#     csv_out = os.path.join(args.out, f"events_{idx}.csv")
#     csv_summary_out = os.path.join(args.out, f"vehicle_summary_{idx}.csv")

#     # Init video properties
#     cap = cv2.VideoCapture(args.video)
#     assert cap.isOpened(), f"Cannot open video: {args.video}"
#     fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
#     width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     cap.release()

#     writer = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

#     # Init modules
#     model = YOLO(args.model)
#     counter = LineCrossCounter((args.line[0], args.line[1]), (args.line[2], args.line[3]))
#     logger = CSVLogger(csv_out, CLASS_LABELS)

#     # FPS tracking
#     t_last = time.time()
#     ema_fps = None
#     frame_idx = 0

#     # Streaming tracker
#     stream = model.track(
#         source=args.video,
#         tracker=args.tracker,
#         conf=args.conf,
#         iou=args.iou,
#         classes=VEHICLE_CLASS_IDS,
#         persist=True,
#         stream=True,
#         verbose=False
#     )

#     for results in stream:
#         frame = results.orig_img
#         frame_idx += 1

#         # FPS update
#         t_now = time.time()
#         inst_fps = 1.0 / max(1e-6, t_now - t_last)
#         t_last = t_now
#         ema_fps = inst_fps if ema_fps is None else 0.9*ema_fps + 0.1*inst_fps

#         # Process detections
#         boxes = results.boxes
#         annotated = results.plot()

#         if boxes is not None and len(boxes) > 0:
#             xyxy = boxes.xyxy.cpu().numpy()
#             clss = boxes.cls.cpu().numpy().astype(int)
#             confs = boxes.conf.cpu().numpy()
#             ids   = boxes.id.cpu().numpy() if boxes.id is not None else np.array([-1]*len(boxes))

#             for (x1, y1, x2, y2), c, conf, tid in zip(xyxy, clss, confs, ids):
#                 obj = {"vehicle_id": int(tid), "bbox": [x1,y1,x2,y2], "class": int(c), "conf": float(conf)}

#                 crossed, direction = (0, "") if tid==-1 else counter.check_cross(obj)
#                 logger.log(frame_idx, fps, obj, crossed, direction)

#                 if tid != -1:
#                     cx, cy = int((x1+x2)/2), int((y1+y2)/2)
#                     cv2.circle(annotated, (cx, cy), 3, (255, 255, 255), -1)
#                     if crossed:
#                         cv2.putText(annotated, direction, (cx+6, cy-6),
#                                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

#         # HUD and overlay
#         draw_line(annotated, (args.line[0], args.line[1]), (args.line[2], args.line[3]),
#                   counter.count_A_to_B, counter.count_B_to_A)
#         put_hud(annotated, ema_fps or inst_fps, os.path.basename(args.model))
#         writer.write(annotated)

#     writer.release()
#     logger.close()
#     print(f"\nSaved annotated video: {video_out}")
#     print(f"Saved CSV: {csv_out}")

#     # Extract vehicle summary
#     summary_df = extract_vehicle_summary(csv_out, csv_summary_out)
#     print(summary_df)



# # ---------------- CLI ----------------
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Vehicle counting and tracking with YOLOv11 + BoT-SORT") # (by default)
#     parser.add_argument("--video", type=str, required=True, help="Path to input video")
#     parser.add_argument("--model", type=str, default="yolo11s.pt", help="YOLO model weights file")
#     parser.add_argument("--tracker", type=str, default="botsort.yaml", help="Tracker config file")
#     parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
#     parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold")
#     parser.add_argument("--line", type=int, nargs=4, default=[200,300,900,300], # default as horizontal line in middle of 1280x720
#                         metavar=("x1","y1","x2","y2"), help="Virtual line coordinates")
#     parser.add_argument("--out", type=str, default="runs/traffic", help="Output directory")
#     args = parser.parse_args()

#     main(args)
