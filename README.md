# T-maze Pipeline

Automated video analysis pipeline for T-maze behavioral experiments with pose estimation, ROI detection, and decision analysis.

## Features

- **Distortion Check**: Charuco board-based lens distortion analysis
- **ROI Inference**: Automatic detection of 23 T-maze keypoints using SLEAP
- **Pose Inference**: Mouse body keypoint tracking (15 keypoints)
- **Decision Analysis**: Depth-based arm entry detection with timing and correctness scoring
- **Gait Analysis**: Stride filtering and metrics extraction
- **Parallel Processing**: Multi-worker batch processing for large datasets

## Installation

```bash
# Clone the repository
git clone https://github.com/LeoMeow123/tmaze-pipeline.git
cd tmaze-pipeline

# Install with uv
uv sync

# Or install with pip
pip install -e .
```

## Quick Start

```bash
# Run full pipeline
tmaze run --input /path/to/videos --output /path/to/results --meta metadata.csv

# Check if videos need undistortion
tmaze check-distortion --input /path/to/videos

# Run ROI inference (T-maze region detection)
tmaze roi-inference --input /path/to/videos --output /path/to/roi_slp

# Convert ROI predictions to YAML polygons
tmaze convert-roi --input /path/to/roi_slp --output /path/to/roi_yml

# Run pose inference (mouse body tracking)
tmaze pose-inference --input /path/to/videos --output /path/to/poses

# Analyze T-maze decisions
tmaze analyze-decisions --videos /path/to/videos --yml /path/to/roi_yml --meta metadata.csv

# Filter gait strides
tmaze analyze-gait --input gait_per_stride.csv --output gait_filtered.csv
```

## Parallel Processing

For large datasets (500+ videos), use parallel processing to speed up analysis.

### Multi-GPU Inference

Run multiple processes on different GPUs simultaneously. Each worker processes a subset of videos.

```bash
# 2 GPUs - pose inference
tmaze pose-inference -i /videos --gpu 0 --worker-id 0 --num-workers 2 &
tmaze pose-inference -i /videos --gpu 1 --worker-id 1 --num-workers 2 &
wait

# 2 GPUs - ROI inference
tmaze roi-inference -i /videos -o /output --gpu 0 --worker-id 0 --num-workers 2 &
tmaze roi-inference -i /videos -o /output --gpu 1 --worker-id 1 --num-workers 2 &
wait
```

| Flag | Description |
|------|-------------|
| `--gpu N` | GPU device ID (shorthand for `--device cuda:N`) |
| `--worker-id K` | Worker ID, 0-indexed |
| `--num-workers M` | Total number of parallel workers |

Videos are partitioned by index: worker K processes videos where `index % M == K`.

### Multi-Core Decision Analysis

Decision analysis is CPU-bound and can use multiple cores:

```bash
# Use 8 CPU cores
tmaze analyze-decisions --videos /path --yml /rois --meta meta.csv --workers 8
```

### Recommended Workflow

```bash
# Step 1: Multi-GPU pose inference
tmaze pose-inference -i /videos --gpu 0 --worker-id 0 --num-workers 2 &
tmaze pose-inference -i /videos --gpu 1 --worker-id 1 --num-workers 2 &
wait

# Step 2: Multi-GPU ROI inference
tmaze roi-inference -i /videos -o /roi_slp --gpu 0 --worker-id 0 --num-workers 2 &
tmaze roi-inference -i /videos -o /roi_slp --gpu 1 --worker-id 1 --num-workers 2 &
wait

# Step 3: Convert ROI to YAML (fast, single process)
tmaze convert-roi -i /roi_slp -o /roi_yml

# Step 4: Multi-core decision analysis
tmaze analyze-decisions --videos /videos --yml /roi_yml --meta meta.csv -w 8
```

## Pipeline Stages

1. **Video Scan** - Discover and validate input videos
2. **Distortion Check** - Analyze charuco boards for lens distortion
3. **Undistortion** - Apply corrections if needed
4. **ROI Inference** - Detect T-maze region keypoints (23 points)
5. **SLP to YAML** - Convert ROI predictions to polygon format
6. **Pose Inference** - Run SLEAP pose estimation (15 mouse keypoints)
7. **Decision Analysis** - Determine arm entry choices using depth-based algorithm
8. **Gait Analysis** - Extract and filter stride metrics

## Decision Algorithm (Depth-Based)

The pipeline uses a depth-based decision algorithm to determine which arm the mouse commits to:

1. **Gate Detection**: Find when mouse leaves segment4 (coverage < 30% for 5+ frames)
2. **Arm Dwell Detection**: Find intervals where body coverage in arm >= 95%
3. **Depth Measurement**: Calculate distance from snout to junction boundary
4. **Commit Selection**: Pick the entry with deepest penetration (> 50px) and hindpaw present

```
Candidate Entries → Filter by depth > 50px AND hindpaw in arm → Select deepest
```

## Data Format

### Input Videos
- Format: MP4 or AVI
- Naming: `Day{N}_{MouseID}_Trial{K}.mp4`
- Example: `Day10_13983_Trial5.mp4`

### Metadata CSV
```csv
day,mouse,reward
DAY10,13983,L
DAY10,13984,R
```

### Output Files

#### events.csv
Per-trial timing events:
```csv
day,mouse,trial,stem,video_path,enter_seg1_frame,enter_seg4_frame,enter_junction_frame,commit_side,commit_frame,ok,enter_seg1_ms,enter_seg4_ms,enter_junction_ms,commit_ms,junction_explore_frames_precommit,junction_explore_ms_precommit,probes_L,probes_R,probe_frames_L,probe_frames_R,coverage_thr_rule
```

#### decisions.csv
Per-trial decision summary:
```csv
video_path,stem,day,mouse,trial,reward,fps,ok,reason,gate_frame,gate_reason,entry_side,entry_frame,entry_ms,entry_threshold,correct_TF,correct_bin
```

#### metrics.csv
Simplified metrics for analysis:
```csv
day,mouse,trial,stem,video_path,choice_LR,latency_choice_ms,stem_latency_ms,junction_explore_ms,reward,correct_TF,correct_bin,probes_L,probes_R,probe_frames_L,probe_frames_R
```

#### gait_filtered.csv
Filtered stride data after confidence and edge filtering.

## Configuration

Key thresholds in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_valid_sec` | 0.5s | Warm-up time before considering arm entries |
| `seg4_leave_threshold` | 30% | Coverage threshold for gate detection |
| `strict_dwell_threshold` | 95% | Body coverage for strict arm entry |
| `soft_dwell_threshold` | 60% | Body coverage for soft arm entry |
| `min_depth_px` | 50px | Minimum depth from junction for commit |
| `gait_confidence_threshold` | 0.3 | Minimum pose confidence for gait |

## Model Paths

Default SLEAP models (configurable):

```python
# ROI models (23 T-maze keypoints)
roi_models = [
    "models/251126_191944.centroid.n=50",
    "models_centered_scale0.5/centered_instance.n=50",
]

# Pose models (15 mouse keypoints)
pose_models = [
    "models_500/251204_003259.centroid.n=503",
    "models_500/centered_instance.n=503",
]
```

## T-maze ROI Keypoints (23)

- **Arms**: arm_right (3), arm_left (3), junction (5)
- **Stem**: segment1 (3), segment2 (3), segment3 (3), segment4 (3)

## Mouse Pose Keypoints (15)

```
snout, mouth, forepawR2, forepawR1, forepawL1, forepawL2,
hindpawR2, hindpawR1, hindpawL2, hindpawL1,
tailbase, tail1, tail2, tail3, tailtip
```

## Requirements

- Python >= 3.12
- SLEAP-io >= 0.2.0
- sleap-nn (for inference)
- OpenCV >= 4.8.0
- NumPy, Pandas, Shapely
- Click, Rich (for CLI)

## License

MIT License - see LICENSE file

## Author

LeoMeow123 (y9li@ucsd.edu)
