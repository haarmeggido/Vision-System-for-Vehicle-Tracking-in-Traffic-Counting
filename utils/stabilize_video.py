from pathlib import Path
import subprocess
import argparse
from tqdm import tqdm


def stabilize_video(video_path: Path, input_dir: Path, output_dir: Path):
    relative = video_path.relative_to(input_dir)

    output_path = output_dir / relative
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        print(f"SKIP {relative}")
        return

    transforms_file = output_path.with_suffix(".trf")

    print(f"PROCESS {relative}")

    # Pass 1 - estimate camera motion
    cmd_detect = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vf",
        f"vidstabdetect=shakiness=5:accuracy=15:result={transforms_file}",
        "-f",
        "null",
        "-"
    ]

    subprocess.run(
        cmd_detect,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True
    )

    # Pass 2 - stabilize and remove audio
    cmd_transform = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vf",
        f"vidstabtransform=input={transforms_file}:zoom=0:smoothing=30",
        "-an",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "18",
        str(output_path)
    ]

    subprocess.run(
        cmd_transform,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True
    )

    transforms_file.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(
        description="Recursively stabilize MP4 videos and remove audio."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input directory containing MP4 files"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for processed videos"
    )

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    video_files = list(input_dir.rglob("*.mp4"))

    print(f"Found {len(video_files)} videos.")

    for video_path in tqdm(video_files):
        try:
            stabilize_video(video_path, input_dir, output_dir)
        except Exception as e:
            print(f"ERROR {video_path}: {e}")

    print("Done.")


if __name__ == "__main__":
    main()