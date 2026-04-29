"""
Gait stride analysis and filtering.

Adapted from: 2025-12-12-GoPro-Tmaz-gait-analysis/01.filter_strides_by_confidence.py

This module filters gait stride data based on confidence thresholds
and removes edge strides from continuous sequences.

Filter tiers (cumulative):
    1. gait_per_stride.csv          — raw strides
    2. gait_per_stride_edge_only.csv — first/last stride per video removed
    3. gait_per_stride_edge_forward.csv — edge + forward-direction only
    4. gait_per_stride_filtered.csv — edge + forward + angular-velocity [-20,20]

Tier 3 is designed for regional gait analysis (stem/junction/arm) where
the angular-velocity filter would remove nearly all turning strides.
"""

from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np

from tmaze_pipeline.config import PipelineConfig


# ---------------------------------------------------------------------------
# Forward-direction filter
# ---------------------------------------------------------------------------

def filter_forward_direction(
    df: pd.DataFrame,
    stride_length_col: str = "stride_length_cm",
) -> pd.DataFrame:
    """Keep only forward-direction strides.

    Forward direction is determined by the sign of the median stride length.
    For GoPro T-maze videos the median is typically positive.  Strides whose
    ``stride_length_cm`` has the opposite sign are walking backward and are
    removed.

    Args:
        df: DataFrame with at least ``stride_length_col``.
        stride_length_col: Column containing signed stride length.

    Returns:
        Filtered copy of *df* with backward strides removed.
    """
    if stride_length_col not in df.columns or len(df) == 0:
        return df.copy()

    median_sl = df[stride_length_col].median()
    forward_sign = np.sign(median_sl)
    if forward_sign == 0:
        return df.copy()
    return df[(df[stride_length_col] * forward_sign) >= 0].copy()


# ---------------------------------------------------------------------------
# Angular-velocity filter
# ---------------------------------------------------------------------------

def filter_angular_velocity(
    df: pd.DataFrame,
    angvel_col: str = "angular_velocity_deg_s_mean",
    lo: float = -20.0,
    hi: float = 20.0,
) -> pd.DataFrame:
    """Keep strides whose mean angular velocity falls within [lo, hi].

    This removes turning strides.  **Not suitable for regional gait analysis**
    (junction/arm) because most strides in those regions involve turns.

    Args:
        df: DataFrame with *angvel_col*.
        angvel_col: Column name for angular velocity.
        lo: Lower bound (inclusive).
        hi: Upper bound (inclusive).

    Returns:
        Filtered copy of *df*.
    """
    if angvel_col not in df.columns or len(df) == 0:
        return df.copy()
    return df[(df[angvel_col] >= lo) & (df[angvel_col] <= hi)].copy()


# ---------------------------------------------------------------------------
# Main entry point — tiered filtering
# ---------------------------------------------------------------------------

def filter_strides(
    stride_csv: Path,
    confidence_csv: Optional[Path] = None,
    output_csv: Path = Path("gait_filtered.csv"),
    confidence_threshold: float = 0.3,
    config: Optional[PipelineConfig] = None,
    verbose: bool = True,
) -> dict:
    """
    Filter gait strides by confidence and remove edge strides.

    Args:
        stride_csv: Input CSV with per-stride gait data
        confidence_csv: Optional CSV with per-video keypoint confidence
        output_csv: Output path for filtered data
        confidence_threshold: Minimum confidence to keep stride
        config: Optional pipeline config
        verbose: Print detailed statistics

    Returns:
        Summary dict with counts and statistics
    """
    if config is None:
        config = PipelineConfig()

    # Load stride data
    df = pd.read_csv(stride_csv)
    original_count = len(df)

    if verbose:
        print(f"\n{'='*60}")
        print("GAIT STRIDE FILTERING")
        print(f"{'='*60}")
        print(f"\n[1/4] Loaded {original_count:,} strides from {stride_csv}")

    # Calculate stride confidence from keypoint confidence
    if confidence_csv and Path(confidence_csv).exists():
        df_conf = pd.read_csv(confidence_csv)
        df = add_stride_confidence(df, df_conf, config.gait_keypoints)
        if verbose:
            mean_conf = df["stride_confidence"].mean()
            median_conf = df["stride_confidence"].median()
            print(f"\n[2/4] Stride confidence stats:")
            print(f"      Mean: {mean_conf:.3f}, Median: {median_conf:.3f}")
    elif "stride_confidence" not in df.columns:
        df["stride_confidence"] = 1.0
        if verbose:
            print(f"\n[2/4] No confidence data - using default 1.0")

    # Filter by confidence threshold
    df = df[df["stride_confidence"] > confidence_threshold].copy()
    after_confidence = len(df)
    removed_conf = original_count - after_confidence
    if verbose:
        pct_removed = 100 * removed_conf / original_count if original_count > 0 else 0
        print(f"\n[3/4] Filtering by confidence > {confidence_threshold}:")
        print(f"      Removed: {removed_conf:,} ({pct_removed:.1f}%)")
        print(f"      Remaining: {after_confidence:,}")

    # Remove first and last stride in each continuous sequence
    df = remove_edge_strides(df)
    after_edges = len(df)
    removed_edges = after_confidence - after_edges
    if verbose:
        pct_edges = 100 * removed_edges / after_confidence if after_confidence > 0 else 0
        print(f"\n[4/4] Removing edge strides:")
        print(f"      Removed: {removed_edges:,} ({pct_edges:.1f}%)")
        print(f"      Final: {after_edges:,}")

    # Save filtered data
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    # Compute summary statistics
    stats = {
        "original": original_count,
        "after_confidence": after_confidence,
        "filtered": after_edges,
        "removed_confidence": removed_conf,
        "removed_edges": removed_edges,
        "total_removed": original_count - after_edges,
        "pct_retained": 100 * after_edges / original_count if original_count > 0 else 0,
    }

    # Per-mouse statistics
    if "mouse" in df.columns and len(df) > 0:
        mouse_counts = df.groupby("mouse").size()
        stats["per_mouse"] = {
            "mean": float(mouse_counts.mean()),
            "median": float(mouse_counts.median()),
            "min": int(mouse_counts.min()),
            "max": int(mouse_counts.max()),
            "n_mice": len(mouse_counts),
        }
        if verbose:
            print(f"\n      Strides per mouse: mean={mouse_counts.mean():.1f}, "
                  f"median={mouse_counts.median():.1f}, range=[{mouse_counts.min()}, {mouse_counts.max()}]")

    # Per-genotype statistics (if available)
    if "Genotype" in df.columns and len(df) > 0:
        genotype_counts = df["Genotype"].value_counts().to_dict()
        stats["per_genotype"] = genotype_counts
        if verbose:
            print(f"\n      Strides by genotype:")
            for gt, cnt in genotype_counts.items():
                print(f"        {gt}: {cnt:,}")

    if verbose:
        print(f"\n{'='*60}")
        print(f"FILTERING COMPLETE")
        print(f"  Original: {original_count:,} strides")
        print(f"  Final: {after_edges:,} strides ({stats['pct_retained']:.1f}% retained)")
        print(f"  Output: {output_csv}")
        print(f"{'='*60}\n")

    return stats


def filter_strides_tiered(
    stride_csv: Path,
    output_dir: Path,
    confidence_csv: Optional[Path] = None,
    confidence_threshold: float = 0.3,
    angvel_range: tuple[float, float] = (-20.0, 20.0),
    config: Optional[PipelineConfig] = None,
    verbose: bool = True,
) -> dict:
    """Produce all four filter tiers from a single raw stride CSV.

    Outputs written to *output_dir*:

    ======================================  ====================================
    File                                    Description
    ======================================  ====================================
    ``gait_per_stride.csv``                 Copy of the raw input
    ``gait_per_stride_edge_only.csv``       First/last stride per video removed
    ``gait_per_stride_edge_forward.csv``    Edge + forward-direction only
    ``gait_per_stride_filtered.csv``        Edge + forward + angular-velocity
    ======================================  ====================================

    Args:
        stride_csv: Input CSV with per-stride gait data.
        output_dir: Directory for all output CSVs.
        confidence_csv: Optional per-video keypoint confidence CSV.
        confidence_threshold: Minimum pose confidence to keep a stride.
        angvel_range: (lo, hi) inclusive bounds for angular-velocity filter.
        config: Optional pipeline config (provides *gait_keypoints*).
        verbose: Print progress and summary counts.

    Returns:
        Dict with stride counts at each tier.
    """
    if config is None:
        config = PipelineConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load raw strides ------------------------------------------------
    df_raw = pd.read_csv(stride_csv)
    n_raw = len(df_raw)

    if verbose:
        print(f"\n{'='*60}")
        print("GAIT STRIDE FILTERING (tiered)")
        print(f"{'='*60}")
        print(f"  Loaded {n_raw:,} raw strides from {stride_csv}")

    # ---- Confidence filter -----------------------------------------------
    if confidence_csv and Path(confidence_csv).exists():
        df_conf = pd.read_csv(confidence_csv)
        df_raw = add_stride_confidence(df_raw, df_conf, config.gait_keypoints)
    elif "stride_confidence" not in df_raw.columns:
        df_raw["stride_confidence"] = 1.0
    df_raw = df_raw[df_raw.get("stride_confidence", 1.0) > confidence_threshold].copy()
    n_after_conf = len(df_raw)

    # ---- Tier 1: raw (post-confidence) -----------------------------------
    df_raw.to_csv(output_dir / "gait_per_stride.csv", index=False)

    # ---- Tier 2: edge removal --------------------------------------------
    df_edge = remove_edge_strides(df_raw)
    n_edge = len(df_edge)
    df_edge.to_csv(output_dir / "gait_per_stride_edge_only.csv", index=False)

    # ---- Tier 3: edge + forward direction --------------------------------
    df_edge_fwd = filter_forward_direction(df_edge)
    n_edge_fwd = len(df_edge_fwd)
    df_edge_fwd.to_csv(output_dir / "gait_per_stride_edge_forward.csv", index=False)

    # ---- Tier 4: edge + forward + angular velocity -----------------------
    df_filtered = filter_angular_velocity(df_edge_fwd, lo=angvel_range[0], hi=angvel_range[1])
    n_filtered = len(df_filtered)
    df_filtered.to_csv(output_dir / "gait_per_stride_filtered.csv", index=False)

    if verbose:
        print(f"  After confidence (>{confidence_threshold}): {n_after_conf:,}")
        print(f"  Tier 2 — edge only:            {n_edge:,}")
        print(f"  Tier 3 — edge + forward:       {n_edge_fwd:,}")
        print(f"  Tier 4 — edge + fwd + angvel:  {n_filtered:,}")
        print(f"  Output dir: {output_dir}")
        print(f"{'='*60}\n")

    return {
        "raw": n_raw,
        "after_confidence": n_after_conf,
        "edge_only": n_edge,
        "edge_forward": n_edge_fwd,
        "filtered": n_filtered,
        "output_dir": str(output_dir),
    }


def add_stride_confidence(
    df_stride: pd.DataFrame,
    df_conf: pd.DataFrame,
    keypoints: tuple[str, ...],
) -> pd.DataFrame:
    """Add stride confidence based on per-video keypoint confidence."""
    # Find available confidence columns
    conf_cols = [f"avg_{kp}" for kp in keypoints if f"avg_{kp}" in df_conf.columns]

    if not conf_cols:
        df_stride["stride_confidence"] = np.nan
        return df_stride

    # Calculate mean confidence across keypoints for each video
    df_conf["stride_confidence"] = df_conf[conf_cols].mean(axis=1, skipna=True)

    # Map to stride data
    if "video_path" in df_conf.columns and "video_path" in df_stride.columns:
        conf_map = df_conf.set_index("video_path")["stride_confidence"].to_dict()
        df_stride["stride_confidence"] = df_stride["video_path"].map(conf_map)
    else:
        df_stride["stride_confidence"] = np.nan

    return df_stride


def remove_edge_strides(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove first and last stride in each continuous sequence.

    A continuous sequence is defined as consecutive stride_ids within a video.
    """
    if "video_path" not in df.columns or "stride_id" not in df.columns:
        return df

    if len(df) == 0:
        return df

    df = df.sort_values(["video_path", "stride_id"]).reset_index(drop=True)

    def process_group(group):
        if len(group) == 0:
            return group

        group = group.sort_values("stride_id").reset_index(drop=True)

        # Identify sequence breaks
        group["stride_diff"] = group["stride_id"].diff()
        group["new_sequence"] = (group["stride_diff"] != 1) | (group["stride_diff"].isna())
        group["sequence_id"] = group["new_sequence"].cumsum()

        # Mark first and last in each sequence
        group["is_first"] = group.groupby("sequence_id").cumcount() == 0
        group["is_last"] = (
            group.groupby("sequence_id").cumcount()
            == group.groupby("sequence_id")["stride_id"].transform("size") - 1
        )

        # Remove edge strides
        to_keep = ~(group["is_first"] | group["is_last"])
        result = group[to_keep].drop(
            columns=["stride_diff", "new_sequence", "sequence_id", "is_first", "is_last"]
        )

        return result

    df = df.groupby("video_path", group_keys=False).apply(process_group, include_groups=False)
    return df.reset_index(drop=True)


def compute_gait_metrics(
    pose_csv: Path,
    output_dir: Path,
    fps: float = 120.0,
) -> dict:
    """
    Compute gait metrics from pose data.

    This is a placeholder for more complex gait analysis.
    The actual implementation would compute:
    - Stride length
    - Stride frequency
    - Stance/swing phases
    - Lateral displacement
    - Body sway

    Args:
        pose_csv: CSV with pose keypoint data
        output_dir: Directory for output files
        fps: Video frame rate

    Returns:
        Summary dict
    """
    # TODO: Implement full gait metric computation
    # This would involve:
    # 1. Detecting paw touchdown/liftoff events
    # 2. Computing stride intervals
    # 3. Calculating stride length, frequency, etc.
    # 4. Computing body sway metrics

    return {"status": "not_implemented"}


def compute_lateral_displacement(
    keypoints: np.ndarray,
    body_axis_pts: tuple[str, str] = ("snout", "tailbase"),
) -> np.ndarray:
    """
    Compute lateral displacement from body axis.

    Args:
        keypoints: T x J x 2 array of keypoint positions
        body_axis_pts: Keypoint names defining body axis

    Returns:
        T-length array of lateral displacement values
    """
    # TODO: Implement lateral displacement calculation
    # This measures how much the mouse deviates from straight-line motion
    return np.zeros(keypoints.shape[0])
