# Vision System for Vehicle Tracking in Traffic Counting

This repository contains the source code for a computer-vision-based system for multi-vehicle identification and tracking in automated traffic counting. The project was developed as part of a thesis on processing road traffic video recordings into structured vehicle-passage events.

## Overview

The system processes an input traffic video and produces event-level traffic-counting outputs. The main pipeline combines:

1. YOLO11s object detection,
2. BoT-SORT multi-object tracking,
3. vehicle crop extraction,
4. ConvNeXt-Base vehicle classification,
5. temporal probability voting with a YOLO compatibility prior,
6. virtual-line crossing detection,
7. event-level safeguards against selected tracker failure modes.

The final output is a set of structured vehicle-passage events containing the detected vehicle category, movement direction, and crossing time/frame.

## Repository Contents

The repository includes the source code, utility scripts, experiment notebooks, and configuration files used to run the developed traffic-counting pipeline.

Large auxiliary files, such as trained model weights, selected datasets, and bulk experiment artifacts, are stored separately and can be downloaded from: [Google Drive repository](https://drive.google.com/drive/folders/1cDZmsxckRprZOdISNckkQrkW9c575v4w?usp=sharing)

## Main Features

- video-based vehicle detection and tracking,
- configurable virtual counting line,
- crop-based vehicle classification into traffic-counting categories,
- temporal aggregation of classification predictions,
- CSV event output,
- metadata JSON output,
- optional annotated video output,
- optional observation/debug CSV output,
- graphical launcher for configuring and running the pipeline.

## Output Files

A typical processing run may generate:

- `events.csv` — accepted vehicle-passage events,
- `metadata.json` — run configuration and processing metadata,
- annotated video file — visualization of detections, tracks, counting line, and counters,
- observation CSV file — optional diagnostic per-frame tracking and classification information.

## Notes

The repository contains the implementation code only. Large files required for reproducing some experiments, including trained classifier checkpoints and selected bulk artifacts, should be downloaded separately using the archive link above.
