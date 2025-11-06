"""Identity event logging and aggregate metrics."""

from __future__ import annotations

import json
import time
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

import numpy as np


def log_identity_event(path: str, event: Dict[str, Any]) -> None:
    """
    Append an identity event to the JSONL log.

    Args:
        path: Destination JSONL file.
        event: Event payload (must be JSON serializable).
    """
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    record = dict(event)
    record.setdefault("ts", time.time())

    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, separators=(",", ":"), default=_json_default) + "\n")


def _json_default(obj):
    """Handle numpy types for JSON serialization."""
    import numpy as np
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def compute_id_metrics(events_path: str) -> Dict[str, Any]:
    """
    Compute high-level metrics from the identity event stream.

    Returns:
        Dict containing id F1, switch rate per minute, and re-attach latency stats.
    """
    path = Path(events_path)
    if not path.exists():
        return {
            "events": 0,
            "id_f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "switch_rate_per_min": 0.0,
            "reattach_latency_mean": None,
            "reattach_latency_p95": None,
        }

    events: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not events:
        return {
            "events": 0,
            "id_f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "switch_rate_per_min": 0.0,
            "reattach_latency_mean": None,
            "reattach_latency_p95": None,
        }

    confirm_reasons = {"join_confirmed", "promote_confirm", "reattach"}
    drop_reasons = {"drop_exit"}
    switch_reasons = {"switch"}

    confirms = [e for e in events if e.get("reason") in confirm_reasons and e.get("label_after") not in (None, "UNK")]
    drops = [e for e in events if e.get("reason") in drop_reasons]
    switches = [e for e in events if e.get("reason") in switch_reasons]

    tp = len(confirms)
    fp = len(switches)
    fn = len(drops)

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    if precision + recall > 0:
        id_f1 = 2 * precision * recall / (precision + recall)
    else:
        id_f1 = 0.0

    frames = [e.get("frame") for e in events if isinstance(e.get("frame"), (int, float))]
    if frames:
        total_frames = max(frames) - min(frames)
    else:
        total_frames = 0
    total_minutes = max(total_frames / (30.0 * 60.0), 1e-6)
    switch_rate = len(switches) / total_minutes

    drop_frame_by_label: Dict[str, int] = {}
    latencies: List[int] = []
    for event in events:
        reason = event.get("reason")
        frame = event.get("frame")
        if not isinstance(frame, (int, float)):
            continue
        label_before = event.get("label_before")
        label_after = event.get("label_after")

        if reason == "drop_exit" and label_before and label_before != "UNK":
            drop_frame_by_label[label_before] = int(frame)
        elif reason == "reattach" and label_after and label_after != "UNK":
            drop_frame = drop_frame_by_label.get(label_after)
            if drop_frame is not None:
                latencies.append(int(frame) - drop_frame)
                del drop_frame_by_label[label_after]

    latency_mean = float(mean(latencies)) if latencies else None
    latency_p95 = float(np.percentile(latencies, 95)) if latencies else None

    return {
        "events": len(events),
        "id_f1": round(id_f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "switch_rate_per_min": round(switch_rate, 4),
        "reattach_latency_mean": latency_mean,
        "reattach_latency_p95": latency_p95,
    }

