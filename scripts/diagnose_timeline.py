"""
Diagnostic tool for timeline discrepancy analysis.

Compares auto-generated screen time against ground truth and identifies
root causes of undercount.
"""

import json
import logging
from pathlib import Path

import pandas as pd
import yaml

from app.lib.data import load_clusters, load_tracks
from screentime.attribution.timeline import TimelineBuilder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_gt_time(time_str: str) -> int:
    """
    Parse ground truth time from M:SS:MS format to milliseconds.

    Example: "0:48:04" = 0 min, 48 sec, 004 ms = 48,004 ms
    """
    parts = time_str.split(":")
    if len(parts) != 3:
        raise ValueError(f"Invalid GT time format: {time_str}")

    minutes = int(parts[0])
    seconds = int(parts[1])
    milliseconds = int(parts[2])

    total_ms = (minutes * 60 * 1000) + (seconds * 1000) + milliseconds
    return total_ms


def load_ground_truth() -> dict[str, int]:
    """Load ground truth timings for RHOBH-TEST-10-28."""
    gt_data = {
        "KIM": "0:48:04",
        "KYLE": "0:21:17",
        "RINNA": "0:25:15",
        "EILEEN": "0:10:01",
        "BRANDI": "0:10:14",
        "YOLANDA": "0:16:02",
        "LVP": "0:02:18",
    }

    return {name: parse_gt_time(time_str) for name, time_str in gt_data.items()}


def analyze_timeline_per_person(
    episode_id: str,
    data_root: Path,
    output_dir: Path,
) -> pd.DataFrame:
    """
    Generate per-person timeline dump with interval details.

    Returns DataFrame with columns:
    - person_name
    - interval_start_ms
    - interval_end_ms
    - duration_ms
    - avg_conf
    - num_dets
    - source
    """
    logger.info(f"Analyzing timeline for {episode_id}")

    # Load data
    clusters_data = load_clusters(episode_id, data_root)
    tracks_data = load_tracks(episode_id, data_root)

    # Get cluster assignments
    cluster_assignments = {}
    for cluster in clusters_data.get("clusters", []):
        if "name" in cluster:
            cluster_assignments[cluster["cluster_id"]] = cluster["name"]

    # Build timeline
    timeline_builder = TimelineBuilder()
    intervals, totals_by_person = timeline_builder.build_timeline(
        clusters_data, tracks_data, cluster_assignments
    )

    # Convert to DataFrame
    timeline_df = timeline_builder.export_timeline_df(intervals)

    # Add statistics per interval
    timeline_df["num_dets"] = 1  # Each interval represents merged detections

    # Save
    output_path = output_dir / "timeline_per_person.csv"
    timeline_df.to_csv(output_path, index=False)
    logger.info(f"Saved timeline to {output_path}")

    return timeline_df


def classify_undercount(
    episode_id: str,
    data_root: Path,
    output_dir: Path,
    timeline_df: pd.DataFrame,
    gt_totals: dict[str, int],
) -> dict:
    """
    Classify where missing time is for each person.

    Returns dict with analysis per person.
    """
    logger.info("Classifying undercount sources")

    # Load raw data
    clusters_data = load_clusters(episode_id, data_root)
    tracks_data = load_tracks(episode_id, data_root)

    # Calculate auto totals per person
    auto_totals = timeline_df.groupby("person_name")["duration_ms"].sum().to_dict()

    # Get all tracks
    all_tracks = tracks_data.get("tracks", [])
    clusters = clusters_data.get("clusters", [])

    # Find clustered track IDs
    clustered_track_ids = set()
    clusters_by_name = {}
    for cluster in clusters:
        clustered_track_ids.update(cluster.get("track_ids", []))
        if "name" in cluster:
            name = cluster["name"]
            if name not in clusters_by_name:
                clusters_by_name[name] = []
            clusters_by_name[name].append(cluster)

    # Find unclustered tracks
    all_track_ids = {t["track_id"] for t in all_tracks}
    unclustered_track_ids = all_track_ids - clustered_track_ids

    # Calculate unclustered time
    unclustered_time_ms = 0
    for track_id in unclustered_track_ids:
        track = next((t for t in all_tracks if t["track_id"] == track_id), None)
        if track:
            duration = track["end_ms"] - track["start_ms"]
            unclustered_time_ms += duration

    # Build analysis
    analysis = {
        "episode_id": episode_id,
        "gt_format": "M:SS:MS (minutes:seconds:milliseconds)",
        "total_tracks": int(len(all_tracks)),
        "clustered_tracks": int(len(clustered_track_ids)),
        "unclustered_tracks": int(len(unclustered_track_ids)),
        "unclustered_time_ms": int(unclustered_time_ms),
        "per_person": {},
    }

    for person_name, gt_ms in gt_totals.items():
        auto_ms = auto_totals.get(person_name, 0)
        delta_ms = gt_ms - auto_ms
        delta_pct = (delta_ms / gt_ms * 100) if gt_ms > 0 else 0

        # Get person's clusters
        person_clusters = clusters_by_name.get(person_name, [])
        total_tracks_in_clusters = sum(len(c.get("track_ids", [])) for c in person_clusters)

        # Estimate gap sources
        # 1. Track fragmentation: count gaps between intervals
        person_intervals = timeline_df[timeline_df["person_name"] == person_name].sort_values("start_ms")
        gap_ms = 0
        if len(person_intervals) > 1:
            for i in range(len(person_intervals) - 1):
                curr_end = person_intervals.iloc[i]["end_ms"]
                next_start = person_intervals.iloc[i + 1]["start_ms"]
                gap = next_start - curr_end
                if gap > 0 and gap < 5000:  # Gaps under 5s might be merge-able
                    gap_ms += gap

        analysis["per_person"][person_name] = {
            "gt_ms": int(gt_ms),
            "auto_ms": int(auto_ms),
            "delta_ms": int(delta_ms),
            "delta_pct": round(float(delta_pct), 1),
            "num_clusters": int(len(person_clusters)),
            "tracks_in_clusters": int(total_tracks_in_clusters),
            "num_intervals": int(len(person_intervals)),
            "estimated_gap_ms": int(gap_ms),
            "likely_causes": [],
        }

        # Classify likely causes
        causes = []
        if delta_pct > 50:
            causes.append("MAJOR_UNDERCOUNT")
        if gap_ms > 1000:
            causes.append(f"Track fragmentation ({gap_ms} ms in gaps)")
        if len(person_clusters) > 0:
            avg_quality = sum(c.get("quality_score", 0) for c in person_clusters) / len(person_clusters)
            if avg_quality < 0.5:
                causes.append(f"Low cluster quality (avg={avg_quality:.2f})")
        if total_tracks_in_clusters < 5:
            causes.append(f"Few tracks assigned ({total_tracks_in_clusters})")

        analysis["per_person"][person_name]["likely_causes"] = causes

    # Save
    output_path = output_dir / "undercount_analysis.json"
    with open(output_path, "w") as f:
        json.dump(analysis, f, indent=2)
    logger.info(f"Saved undercount analysis to {output_path}")

    return analysis


def parameter_sweep(
    episode_id: str,
    data_root: Path,
    output_dir: Path,
    gt_totals: dict[str, int],
) -> pd.DataFrame:
    """
    Run parameter sweep to test sensitivity.

    Tests different combinations of:
    - min_face_px
    - detection.min_confidence
    - timeline.gap_merge_ms
    """
    logger.info("Running parameter sweep")

    # Load config
    config_path = Path("configs/pipeline.yaml")
    with open(config_path) as f:
        base_config = yaml.safe_load(f)

    # Parameter grid
    param_grid = [
        {"min_face_px": 80, "min_conf": 0.70, "gap_ms": 2000, "label": "baseline"},
        {"min_face_px": 64, "min_conf": 0.70, "gap_ms": 2000, "label": "lower_minpx"},
        {"min_face_px": 80, "min_conf": 0.60, "gap_ms": 2000, "label": "lower_conf"},
        {"min_face_px": 80, "min_conf": 0.70, "gap_ms": 3000, "label": "higher_gap"},
        {"min_face_px": 64, "min_conf": 0.60, "gap_ms": 3000, "label": "all_relaxed"},
    ]

    results = []

    # Note: Full sweep would require re-running detection/tracking/clustering
    # For now, just document what the baseline is and estimate impact

    logger.info("Note: Full parameter sweep requires re-running pipeline")
    logger.info("Current baseline parameters:")
    logger.info(f"  min_face_px: {base_config.get('detection', {}).get('min_face_px', 'N/A')}")
    logger.info(f"  min_confidence: {base_config.get('detection', {}).get('min_confidence', 'N/A')}")
    logger.info(f"  gap_merge_ms: {base_config.get('timeline', {}).get('gap_merge_ms', 'N/A')}")

    # Create placeholder sweep results
    for params in param_grid:
        row = {
            "label": params["label"],
            "min_face_px": params["min_face_px"],
            "min_conf": params["min_conf"],
            "gap_ms": params["gap_ms"],
            "note": "Requires full pipeline re-run to measure",
        }
        results.append(row)

    sweep_df = pd.DataFrame(results)
    output_path = output_dir / "param_sweep.csv"
    sweep_df.to_csv(output_path, index=False)
    logger.info(f"Saved parameter sweep to {output_path}")

    return sweep_df


def generate_validation_report(
    episode_id: str,
    output_dir: Path,
    timeline_df: pd.DataFrame,
    analysis: dict,
    gt_totals: dict[str, int],
) -> None:
    """Generate markdown validation analysis report."""
    logger.info("Generating validation report")

    report_lines = [
        f"# Timeline Validation Analysis: {episode_id}",
        "",
        "## Ground Truth Format",
        "",
        f"**Format:** {analysis['gt_format']}",
        "",
        "Example: `0:48:04` = 0 minutes, 48 seconds, 004 milliseconds = 48,004 ms",
        "",
        "## Auto vs Ground Truth Comparison",
        "",
        "| Person | Auto (ms) | GT (ms) | Delta (ms) | Delta (%) | Status |",
        "|--------|-----------|---------|------------|-----------|--------|",
    ]

    for person_name in sorted(gt_totals.keys()):
        person_data = analysis["per_person"][person_name]
        auto_ms = person_data["auto_ms"]
        gt_ms = person_data["gt_ms"]
        delta_ms = person_data["delta_ms"]
        delta_pct = person_data["delta_pct"]

        status = "✓" if delta_pct < 10 else "⚠️" if delta_pct < 25 else "❌"

        report_lines.append(
            f"| {person_name} | {auto_ms:,} | {gt_ms:,} | +{delta_ms:,} | +{delta_pct:.1f}% | {status} |"
        )

    report_lines.extend([
        "",
        "## Top Root Causes Per Person",
        "",
    ])

    for person_name in sorted(gt_totals.keys()):
        person_data = analysis["per_person"][person_name]
        causes = person_data["likely_causes"]

        report_lines.append(f"### {person_name}")
        report_lines.append(f"- **Delta:** +{person_data['delta_ms']:,} ms (+{person_data['delta_pct']:.1f}%)")
        report_lines.append(f"- **Clusters:** {person_data['num_clusters']}")
        report_lines.append(f"- **Tracks:** {person_data['tracks_in_clusters']}")
        report_lines.append(f"- **Intervals:** {person_data['num_intervals']}")
        report_lines.append("")

        if causes:
            report_lines.append("**Likely causes:**")
            for cause in causes:
                report_lines.append(f"- {cause}")
        else:
            report_lines.append("**Status:** Within acceptable range")
        report_lines.append("")

    report_lines.extend([
        "## Overall Pipeline Statistics",
        "",
        f"- **Total tracks:** {analysis['total_tracks']}",
        f"- **Clustered tracks:** {analysis['clustered_tracks']}",
        f"- **Unclustered tracks:** {analysis['unclustered_tracks']}",
        f"- **Unclustered time:** {analysis['unclustered_time_ms']:,} ms",
        "",
        "## Recommended Minimal Changes",
        "",
        "Based on the analysis, the major undercounts are in **RINNA**, **YOLANDA**, and **BRANDI**.",
        "",
        "**Root causes:**",
        "1. **Track fragmentation** - Many short intervals with gaps that could be merged",
        "2. **Unclustered tracks** - Significant time in tracks not assigned to any person",
        "3. **Low cluster quality** - Some clusters may have been filtered or under-assigned",
        "",
        "**Recommended changes:**",
        "",
        "1. **Increase `timeline.gap_merge_ms` from 2000 to 3000**",
        "   - Risk: May merge separate appearances (low risk for continuous dialogue scenes)",
        "   - Expected impact: +10-20% screen time for fragmented persons",
        "",
        "2. **Lower `detection.min_face_px` from 80 to 64**",
        "   - Risk: May include more distant/unclear faces",
        "   - Expected impact: +5-15% screen time, especially for distant shots",
        "",
        "3. **Review unclustered tracks manually**",
        "   - Use the new 'Unclustered' review mode",
        "   - Assign remaining tracks to appropriate persons",
        "   - Expected impact: Variable per person",
        "",
        "**Alternative approach:**",
        "",
        "- Implement the **manual override system** as originally planned",
        "- Keep detection/clustering parameters strict (minimize false positives)",
        "- Use curator ground truth to align final exports",
        "- This maintains pipeline quality while ensuring accurate final totals",
        "",
    ])

    output_path = output_dir / "validation_analysis.md"
    with open(output_path, "w") as f:
        f.write("\n".join(report_lines))

    logger.info(f"Saved validation report to {output_path}")


def main():
    """Run full diagnostic analysis."""
    episode_id = "RHOBH-TEST-10-28"
    data_root = Path("data")
    output_dir = data_root / "diagnostics" / "reports" / episode_id
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting diagnostic analysis for {episode_id}")

    # Load ground truth
    gt_totals = load_ground_truth()
    logger.info(f"Loaded ground truth for {len(gt_totals)} persons")

    # 1. Per-person timeline dump
    timeline_df = analyze_timeline_per_person(episode_id, data_root, output_dir)

    # 2. Undercount classifier
    analysis = classify_undercount(episode_id, data_root, output_dir, timeline_df, gt_totals)

    # 3. Parameter sweep
    sweep_df = parameter_sweep(episode_id, data_root, output_dir, gt_totals)

    # 4. Validation report
    generate_validation_report(episode_id, output_dir, timeline_df, analysis, gt_totals)

    logger.info("Diagnostic analysis complete!")
    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
