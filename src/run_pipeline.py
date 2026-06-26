import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2
import torch

from traffic_counter.config import (
    DEFAULT_YOLO_CLASSES, 
    PipelineConfig,
    SUPPORTED_CLASSIFIER_ARCHS,
)
from traffic_counter.classification.gddkia_classifier import GDDKIAClassifier
from traffic_counter.classification.voting import create_vote_buffer
from traffic_counter.storage.csv_export import EventCSVWriter
from traffic_counter.tracking.ultralytics_tracker import UltralyticsTracker
from traffic_counter.visualization.annotations import draw_line, draw_track
from traffic_counter.events.line_crossing import LineCrossCounter
from traffic_counter.storage.metadata import build_run_metadata, save_run_metadata
from traffic_counter.storage.observation_debug import ObservationDebugWriter
from traffic_counter.events.event_deduplication import (
    CrossingCandidate,
    CrossingEventDeduplicator,
)

def make_run_paths(out_dir: Path, video_path: Path):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_name = video_path.stem
    run_id = f"{timestamp}_{video_name}"

    video_dir = out_dir / "video"
    csv_dir = out_dir / "vehicle_summary"
    metadata_dir = out_dir / "metadata"
    observation_dir = out_dir / "observations"

    video_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    observation_dir.mkdir(parents=True, exist_ok=True)

    video_out_path = video_dir / f"annotated_{run_id}.mp4"
    csv_out_path = csv_dir / f"summary_{run_id}.csv"
    metadata_out_path = metadata_dir / f"run_{run_id}.json"
    observation_out_path = observation_dir / f"observations_{run_id}.csv"


    return run_id, video_out_path, csv_out_path, metadata_out_path, observation_out_path

def crop_bbox(frame, bbox, pad: int = 10):
    x1, y1, x2, y2 = map(int, bbox)
    h, w = frame.shape[:2]

    x1p = max(0, x1 - pad)
    y1p = max(0, y1 - pad)
    x2p = min(w, x2 + pad)
    y2p = min(h, y2 + pad)

    return frame[y1p:y2p, x1p:x2p]


def open_video_properties(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()
    return fps, width, height


def process_video(config: PipelineConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_id, video_out_path, csv_out_path, metadata_out_path, observation_out_path = make_run_paths(
        config.out_dir,
        config.video,
    )

    print(f"--- Run ID: {run_id} ---")
    print(f"Device:       {device}")
    print(f"Video input:  {config.video}")

    if config.save_video:
        print(f"Video output: {video_out_path}")
    else:
        print("Video output: disabled")

    print(f"CSV output:   {csv_out_path}")
    print(f"Metadata output: {metadata_out_path}")
    if config.save_observations:
        print(f"Observation output: {observation_out_path}")
    else:
        print("Observation output: disabled")
    print(f"Voting:      {config.voting_method}")

    fps, width, height = open_video_properties(config.video)

    writer = None
    if config.save_video:
        writer = cv2.VideoWriter(
            str(video_out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

    observation_writer = None
    if config.save_observations:
        observation_writer = ObservationDebugWriter(
            observation_out_path,
            run_id=run_id,
        )
    
    classifier = GDDKIAClassifier(
        model_path=config.classifier_weights,
        device=device,
        arch=config.classifier_arch,
    )

    voter = create_vote_buffer(
        name=config.voting_method,
        classifier=classifier,
        classify_interval=config.classify_interval,
    )

    line = config.line
    event_engine = LineCrossCounter(
        p1=(line[0], line[1]),
        p2=(line[2], line[3]),
        reference_point=config.line_reference,
        allow_same_id_recount=config.allow_same_id_recount,
        same_id_recount_cooldown=config.same_id_recount_cooldown,
    )
        
    event_deduplicator = CrossingEventDeduplicator(
        iou_threshold= config.duplicate_event_iou_threshold,
        center_distance_threshold= config.duplicate_event_center_distance,
        frame_window= config.duplicate_event_frame_window,
    )

    tracker = UltralyticsTracker(
        model_path=config.yolo_model,
        tracker_config=config.tracker,
        classes=config.yolo_classes,
        conf=config.conf,
        iou=config.iou,
    )

    frame_idx = 0
    track_first_seen = {}
    track_last_seen = {}
    duplicate_events_suppressed = 0


    start_time = time.time()

    print(
        f"Starting processing... "
        f"(Target: {config.stop_frame if config.stop_frame > 0 else 'End'})"
    )

    try:
        with EventCSVWriter(csv_out_path, run_id=run_id) as csv_writer:
            for results in tracker.stream(config.video):

                if config.stop_frame > 0 and frame_idx >= config.stop_frame:
                    break

                frame_idx += 1

                frame = results.orig_img
                annotated = frame.copy()
                boxes = results.boxes
                crossing_candidates = []
                frame_observations = []

                if boxes is not None and boxes.id is not None:
                    xyxy = boxes.xyxy.cpu().numpy()
                    ids = boxes.id.cpu().numpy().astype(int)
                    clss = boxes.cls.cpu().numpy().astype(int)
                    confs = boxes.conf.cpu().numpy()

                    for box, track_id, yolo_cls, yolo_conf in zip(xyxy, ids, clss, confs):
                        crop = crop_bbox(frame, box, pad=10)

                        label = voter.observe_crop(
                            track_id, 
                            crop,
                            yolo_class = int(yolo_cls),
                            yolo_confidence = float(yolo_conf),
                        )
                        
                        tid = int(track_id)

                        crossed, direction, event_debug = event_engine.update(
                            tid,
                            box,
                            frame_idx=frame_idx,
                        )
                        category_confidence = voter.current_confidence(tid)
                        category_scores = voter.current_scores(tid)

                        track_first_seen.setdefault(tid, frame_idx)
                        track_age_frames = frame_idx - track_first_seen[tid] + 1

                        previous_seen_frame = track_last_seen.get(tid)
                        frames_since_last_seen = (
                            0 if previous_seen_frame is None else frame_idx - previous_seen_frame
                        )
                        track_last_seen[tid] = frame_idx

                        event_debug["track_age_frames"] = track_age_frames
                        event_debug["frames_since_last_seen"] = frames_since_last_seen

                        if hasattr(voter, "current_yolo_summary"):
                            yolo_summary = voter.current_yolo_summary(tid)
                        else:
                            yolo_summary = {}

                        if observation_writer is not None:
                            frame_observations.append(
                                dict(
                                    frame_idx=frame_idx,
                                    time_sec=frame_idx / fps if fps > 0 else 0,
                                    track_id=tid,
                                    yolo_class=int(yolo_cls),
                                    yolo_confidence=float(yolo_conf),
                                    bbox=box,
                                    event_debug=event_debug,
                                    current_label=label,
                                    category_confidence=category_confidence,
                                    category_scores=category_scores,
                                    yolo_summary=yolo_summary,
                                )
                            )

                        draw_track(
                            frame=annotated,
                            bbox=box,
                            track_id=tid,
                            label=label,
                            crossed=crossed,
                        )

                        if crossed:
                            crossing_candidates.append(
                                CrossingCandidate(
                                    frame_idx=frame_idx,
                                    track_id=tid,
                                    bbox=box,
                                    direction=direction,
                                    label=label,
                                    category_confidence=category_confidence,
                                    yolo_confidence=float(yolo_conf),
                                    track_age_frames=track_age_frames,
                                    event_debug=event_debug,
                                )
                            )

                    if config.suppress_duplicate_events:
                        accepted_events, suppressed_events = event_deduplicator.filter_frame_candidates(
                            crossing_candidates
                        )
                    else:
                        accepted_events, suppressed_events = crossing_candidates, []

                    for event in suppressed_events:
                        event_engine.undo_count(event.track_id, event.direction, frame_idx=event.frame_idx)
                        duplicate_events_suppressed += 1
                        print(
                            f"[SUPPRESSED DUPLICATE] Frame {event.frame_idx}: "
                            f"Vehicle {event.track_id} suppressed as duplicate crossing "
                            f"({event.label}, conf={event.category_confidence:.3f}, dir={event.direction})"
                        )

                    for event in accepted_events:
                        print(
                            f"[EVENT] Frame {event.frame_idx}: "
                            f"Vehicle {event.track_id} ({event.label}, conf={event.category_confidence:.3f}) "
                            f"moved {event.direction}"
                        )
                        csv_writer.write_event(
                            frame_idx=event.frame_idx,
                            track_id=event.track_id,
                            label=event.label,
                            direction=event.direction,
                            category_confidence=event.category_confidence,
                        )
                    if observation_writer is not None:
                        for observation in frame_observations:
                            observation_writer.write_observation(**observation)

                draw_line(
                    annotated,
                    (line[0], line[1]),
                    (line[2], line[3]),
                    event_engine.stats["A->B"],
                    event_engine.stats["B->A"],
                )

                if writer is not None:
                    writer.write(annotated)

                if frame_idx % 100 == 0:
                    elapsed = time.time() - start_time
                    print(f"Processed {frame_idx} frames in {elapsed:.1f}s...")
    finally:
        if observation_writer is not None:
            observation_writer.close()

    if writer is not None:
        writer.release()
    elapsed = time.time() - start_time
    video_seconds = frame_idx / fps if fps > 0 else 0
    real_time_factor = elapsed / video_seconds if video_seconds > 0 else float("inf")


    metadata = build_run_metadata(
        run_id=run_id,
        config=config,
        video_out_path=video_out_path if config.save_video else None,
        csv_out_path=csv_out_path,
        observation_csv_path=observation_out_path if config.save_observations else None,
        fps=fps,
        width=width,
        height=height,
        frames_processed=frame_idx,
        elapsed_seconds=elapsed,
        video_seconds=video_seconds,
        real_time_factor=real_time_factor,
        event_counts={
            **event_engine.stats,
            "same_id_recount_events": getattr(event_engine, "same_id_recount_events", 0),
            "duplicate_events_suppressed": duplicate_events_suppressed,
        },
        device=str(device),
    )

    save_run_metadata(metadata, metadata_out_path)

    print("Processing finished.")
    print(f"Frames processed:   {frame_idx}")
    print(f"Elapsed time:       {elapsed:.2f}s")
    print(f"Video duration run: {video_seconds:.2f}s")
    print(f"Real-time factor:   {real_time_factor:.3f}")
    print(f"Results in:         {config.out_dir}")

    return metadata


def parse_args():
    parser = argparse.ArgumentParser(
        description="Modular traffic-counting pipeline with YOLO tracking and GDDKiA classification."
    )

    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("../yolo11s.pt"))
    parser.add_argument("--clf_weights", type=Path, required=True)
    parser.add_argument("--tracker", type=str, default="../botsort.yaml")

    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        default=DEFAULT_YOLO_CLASSES,
        help="YOLO class IDs to track",
    )

    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)

    parser.add_argument(
        "--line",
        type=int,
        nargs=4,
        default=[400, 100, 1800, 800],
        metavar=("x1", "y1", "x2", "y2"),
    )

    parser.add_argument("--out", type=Path, default=Path("../runs/traffic"))
    parser.add_argument("--stop_frame", type=int, default=0)
    parser.add_argument("--classify_interval", type=int, default=5)

    parser.add_argument(
        "--classifier_arch",
        type=str,
        default="convnext_small",
        choices=SUPPORTED_CLASSIFIER_ARCHS,
        help="Classifier architecture matching the checkpoint folder/model",    
    )

    parser.add_argument(
        "--save_video",
        action="store_true",
        help="Save annotated output video"
    )

    parser.add_argument(
        "--voting",
        type=str,
        default="majority",
        choices=["majority", "probability", "probability_yolo_prior"],
        help="Track-level classification voting strategy",
    )

    parser.add_argument(
        "--line_reference",
        type=str,
        default="centroid",
        choices=["centroid", "bottom_center"],
        help="Point of the bounding box used for line-crossing detection",    
    )
    
    parser.add_argument(
        "--save_observations",
        action="store_true",
        help="Save per-frame observation data to CSV (for debugging/analysis)"
    )
    
    parser.add_argument(
        "--allow_same_id_recount",
        action="store_true",
        help=(
            "Allow the same tracker ID to generate another crossing event "
            "after a cooldown if it crosses in the opposite direction. "
            "Useful for tracker ID hijack/reuse cases."
        ),
    )

    parser.add_argument(
        "--same_id_recount_cooldown",
        type=int,
        default=30,
        help="Minimum number of frames before the same track ID may be counted again",
    )

    parser.add_argument(
        "--suppress_duplicate_events",
        action="store_true",
        help="Suppress duplicate crossing events from overlapping boxes/tracks",
    )

    parser.add_argument(
        "--duplicate_event_iou_threshold",
        type=float,
        default=0.85,
    )

    parser.add_argument(
        "--duplicate_event_frame_window",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--duplicate_event_center_distance",
        type=float,
        default=30.0,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    config = PipelineConfig(
        video=args.video,
        yolo_model=args.model,
        classifier_weights=args.clf_weights,
        tracker=args.tracker,
        yolo_classes=args.classes,
        conf=args.conf,
        iou=args.iou,
        line=tuple(args.line),
        out_dir=args.out,
        stop_frame=args.stop_frame,
        classify_interval=args.classify_interval,
        classifier_arch=args.classifier_arch,
        save_video=args.save_video,
        voting_method=args.voting,
        line_reference=args.line_reference,
        save_observations=args.save_observations,
        allow_same_id_recount=args.allow_same_id_recount,
        same_id_recount_cooldown=args.same_id_recount_cooldown,
        suppress_duplicate_events=args.suppress_duplicate_events,
        duplicate_event_iou_threshold=args.duplicate_event_iou_threshold,
        duplicate_event_frame_window=args.duplicate_event_frame_window,
        duplicate_event_center_distance=args.duplicate_event_center_distance,
    )

    process_video(config)


if __name__ == "__main__":
    main()