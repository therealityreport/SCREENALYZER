import numpy as np
import pytest

from screenalyzer.reid.assigner import HysteresisAssigner, ReIdConfig
from screenalyzer.reid.faiss_index import FaceIndex
from screenalyzer.reid.metrics import compute_id_metrics, log_identity_event


def _vec(values: tuple[float, float, float]) -> np.ndarray:
    return np.array(values, dtype=np.float32)


BASE_VECTORS = {
    "KYLE": _vec((1.0, 0.0, 0.0)),
    "LISA": _vec((0.0, 1.0, 0.0)),
    "ERIKA": _vec((0.0, 0.0, 1.0)),
}


def _embedding(label: str, frame: int) -> np.ndarray:
    base = BASE_VECTORS[label]
    offset_scale = (frame % 5) * 0.001
    if label == "KYLE":
        offset = _vec((0.0, offset_scale, 0.0))
    elif label == "LISA":
        offset = _vec((offset_scale, 0.0, 0.0))
    else:
        offset = _vec((0.0, offset_scale, offset_scale))
    vec = base + offset
    return vec / np.linalg.norm(vec)


def test_persistent_reid_flow(tmp_path):
    index = FaceIndex(metric="ip", normalize=True)
    for name, vec in BASE_VECTORS.items():
        index.add(name, vec)

    cfg = ReIdConfig(
        tau_join=0.72,
        tau_stay=0.65,
        tau_spawn=0.5,
        ema_window=1,
        confirm_frames=2,
        drop_frames=2,
        ema_alpha=0.5,
        max_sigma=3.0,
    )
    assigner = HysteresisAssigner(index, cfg)

    sequences = [
        {"track": "1", "label": "KYLE", "start": 0, "end": 14},
        {"track": "2", "label": "LISA", "start": 0, "end": 18},
        {"track": "3", "label": "ERIKA", "start": 5, "end": 30},
        {"track": "4", "label": "KYLE", "start": 20, "end": 35},
        {"track": "5", "label": "LISA", "start": 22, "end": 40},
    ]

    events_path = tmp_path / "reid_events.jsonl"
    session_events = []
    assignment_history: dict[str, list[str]] = {}

    for frame in range(0, 45):
        active = [seq for seq in sequences if seq["start"] <= frame <= seq["end"]]
        for seq in active:
            emb = _embedding(seq["label"], frame)
            label, debug = assigner.assign(emb, frame, seq["track"])
            assignment_history.setdefault(seq["track"], []).append(label)
            if "reason" in debug:
                event = dict(debug)
                event.setdefault("frame", frame)
                event.setdefault("track_id", seq["track"])
                event["episode_id"] = "TEST_EP"
                log_identity_event(events_path, event)
                session_events.append(event)

        ended = [seq for seq in sequences if seq["end"] == frame]
        for seq in ended:
            assigner.end_track(seq["track"])

    metrics = compute_id_metrics(events_path)

    assert metrics["switch_rate_per_min"] == pytest.approx(0.0)
    assert metrics["id_f1"] >= 0.9
    assert metrics["events"] >= 4

    assert assignment_history["4"][-1] == "KYLE"
    assert assignment_history["5"][-1] == "LISA"

    faiss_path = tmp_path / "faces.faiss"
    centroids_path = tmp_path / "centroids.json"
    assigner.index.save(str(faiss_path), str(centroids_path))

    restored = FaceIndex(metric="ip", normalize=True)
    restored.load(str(faiss_path), str(centroids_path))
    results = restored.search(BASE_VECTORS["KYLE"], k=1)

    assert results
    assert results[0][0] == "KYLE"
