import argparse
import csv
from pathlib import Path

from traffic_counter.config import (
    DEFAULT_YOLO_CLASSES,
    PipelineConfig,
    SUPPORTED_CLASSIFIER_ARCHS,
)
from run_pipeline import process_video


def find_available_model_dirs(model_root: Path, requested_archs: list[str] | None = None):
    if requested_archs:
        archs = requested_archs
    else:
        archs = SUPPORTED_CLASSIFIER_ARCHS

    available = []

    for arch in archs:
        weights_path = model_root / arch / "best_model.pth"
        if weights_path.exists():
            available.append((arch, weights_path))
        else:
            print(f"[SKIP] Missing weights for {arch}: {weights_path}")

    return available


def write_benchmark_summary(rows: list[dict], output_csv: Path):
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "model",
        "run_id",
        "voting_method",
        "frames_processed",
        "elapsed_seconds",
        "video_seconds",
        "real_time_factor",
        "processed_fps",
        "events_A_to_B",
        "events_B_to_A",
        "events_total",
        "classifier_weights",
        "out_dir",
    ]

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def metadata_to_row(metadata: dict):
    run = metadata["run"]
    processing = metadata["processing"]
    models = metadata["models"]
    outputs = metadata["outputs"]

    return {
        "model": models["classifier"]["architecture"],
        "run_id": run["run_id"],
        "voting_method": processing["voting_method"],
        "frames_processed": processing["frames_processed"],
        "elapsed_seconds": processing["elapsed_seconds"],
        "video_seconds": processing["video_seconds"],
        "real_time_factor": processing["real_time_factor"],
        "processed_fps": processing["processed_fps"],
        "events_A_to_B": processing["events_A_to_B"],
        "events_B_to_A": processing["events_B_to_A"],
        "events_total": processing["events_total"],
        "classifier_weights": models["classifier"]["weights"],
        "out_dir": str(Path(outputs["summary_csv"]).parent.parent),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark all trained classifier architectures inside the full traffic pipeline."
    )

    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("yolo11s.pt"))
    parser.add_argument("--tracker", type=str, default="botsort.yaml")
    parser.add_argument("--model_root", type=Path, required=True)

    parser.add_argument(
        "--archs",
        type=str,
        nargs="+",
        default=None,
        help="Optional subset of classifier architectures to benchmark",
    )

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
        required=True,
        metavar=("x1", "y1", "x2", "y2"),
    )

    parser.add_argument("--out", type=Path, default=Path("runs/pipeline_classifier_benchmark"))
    parser.add_argument("--stop_frame", type=int, default=0)
    parser.add_argument("--classify_interval", type=int, default=5)

    parser.add_argument(
        "--voting",
        type=str,
        default="probability_yolo_prior",
        choices=["majority", "probability", "probability_yolo_prior"],
    )

    parser.add_argument(
        "--save_video",
        action="store_true",
        help="Save annotated videos for every benchmarked model. Usually disabled for speed benchmarks.",
    )

    parser.add_argument(
        "--summary_csv",
        type=Path,
        default=None,
        help="Optional path for benchmark summary CSV",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    model_entries = find_available_model_dirs(
        model_root=args.model_root,
        requested_archs=args.archs,
    )

    if not model_entries:
        raise FileNotFoundError(f"No usable model folders found in {args.model_root}")

    benchmark_rows = []

    print("=" * 70)
    print("PIPELINE CLASSIFIER BENCHMARK")
    print("=" * 70)
    print(f"Video:       {args.video}")
    print(f"Model root:  {args.model_root}")
    print(f"Voting:      {args.voting}")
    print(f"Stop frame:  {args.stop_frame if args.stop_frame > 0 else 'End'}")
    print(f"Models:      {[arch for arch, _ in model_entries]}")
    print("=" * 70)

    for arch, weights_path in model_entries:
        print("\n" + "=" * 70)
        print(f"BENCHMARKING CLASSIFIER: {arch}")
        print("=" * 70)

        # Each classifier gets its own output folder for clarity.
        model_out_dir = args.out / arch

        config = PipelineConfig(
            video=args.video,
            yolo_model=args.model,
            classifier_weights=weights_path,
            tracker=args.tracker,
            yolo_classes=args.classes,
            conf=args.conf,
            iou=args.iou,
            line=tuple(args.line),
            out_dir=model_out_dir,
            stop_frame=args.stop_frame,
            classify_interval=args.classify_interval,
            classifier_arch=arch,
            save_video=args.save_video,
            voting_method=args.voting,
        )

        metadata = process_video(config)
        benchmark_rows.append(metadata_to_row(metadata))

    if args.summary_csv is None:
        summary_csv = args.out / "pipeline_classifier_benchmark_summary.csv"
    else:
        summary_csv = args.summary_csv

    write_benchmark_summary(benchmark_rows, summary_csv)

    print("\n" + "=" * 70)
    print("BENCHMARK FINISHED")
    print("=" * 70)
    print(f"Summary CSV: {summary_csv}")


if __name__ == "__main__":
    main()