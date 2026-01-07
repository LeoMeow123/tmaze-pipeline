"""
T-maze Pipeline - Automated video analysis for T-maze behavioral experiments.

This package provides a complete pipeline for:
1. Video distortion checking (charuco board analysis)
2. Pose estimation inference (SLEAP)
3. ROI detection and labeling
4. T-maze decision analysis (arm entry, dwell times)
5. Gait analysis (stride metrics)

Usage:
    tmaze run --input /path/to/videos --output /path/to/results
    tmaze check-distortion --input /path/to/videos
    tmaze analyze-decisions --input /path/to/data --meta /path/to/meta.csv
"""

__version__ = "0.1.0"
__author__ = "LeoMeow123"

from tmaze_pipeline.config import PipelineConfig

__all__ = ["PipelineConfig", "__version__"]
