import numpy as np

from screenalyzer.reid.assigner import HysteresisAssigner, ReIdConfig
from screenalyzer.reid.faiss_index import FaceIndex


def _vec(*values: float) -> np.ndarray:
    return np.array(values, dtype=np.float32)


def _make_assigner(labels: dict[str, np.ndarray], config: ReIdConfig) -> HysteresisAssigner:
    index = FaceIndex(metric="ip", normalize=True)
    for label, emb in labels.items():
        index.add(label, emb)
    return HysteresisAssigner(index, config)


def test_assign_confirm_known_identity():
    cfg = ReIdConfig(
        tau_join=0.7,
        tau_stay=0.6,
        tau_spawn=0.5,
        ema_window=1,
        confirm_frames=1,
        drop_frames=3,
        ema_alpha=0.5,
        max_sigma=3.0,
    )
    assigner = _make_assigner({"KYLE": _vec(1.0, 0.0, 0.0)}, cfg)

    label, debug = assigner.assign(_vec(0.95, 0.05, 0.0), frame_idx=0, track_id="1")
    assert label == "KYLE"
    assert debug["reason"] == "join_confirmed"
    assert assigner.state["1"].status == "confirmed"


def test_assign_drop_to_unknown():
    cfg = ReIdConfig(
        tau_join=0.7,
        tau_stay=0.65,
        tau_spawn=0.5,
        ema_window=1,
        confirm_frames=1,
        drop_frames=2,
        ema_alpha=0.5,
        max_sigma=3.0,
    )
    assigner = _make_assigner({"KYLE": _vec(1.0, 0.0, 0.0)}, cfg)

    assigner.assign(_vec(1.0, 0.0, 0.0), frame_idx=0, track_id="1")
    assigner.assign(_vec(0.0, 1.0, 0.0), frame_idx=1, track_id="1")
    label, debug = assigner.assign(_vec(0.0, 1.0, 0.0), frame_idx=2, track_id="1")

    assert label == "UNK"
    assert debug["reason"] == "drop_exit"
    assert assigner.state["1"].status == "anonymous"


def test_provisional_promotes_to_confirmed():
    cfg = ReIdConfig(
        tau_join=0.75,
        tau_stay=0.7,
        tau_spawn=0.6,
        ema_window=1,
        confirm_frames=2,
        drop_frames=3,
        ema_alpha=0.4,
        max_sigma=3.0,
    )
    assigner = _make_assigner({"LISA": _vec(0.0, 1.0, 0.0)}, cfg)

    label, debug = assigner.assign(_vec(0.76, 0.65, 0.0), frame_idx=0, track_id="2")
    assert label.startswith("PROV_")
    assert debug["reason"] == "assign_provisional"

    assigner.assign(_vec(0.1, 0.99, 0.0), frame_idx=1, track_id="2")
    label_after, debug_after = assigner.assign(_vec(0.05, 1.0, 0.0), frame_idx=2, track_id="2")

    assert label_after == "LISA"
    assert debug_after["reason"] == "promote_confirm"


def test_spawn_anonymous_label_when_no_match():
    cfg = ReIdConfig(
        tau_join=0.75,
        tau_stay=0.7,
        tau_spawn=0.6,
        ema_window=1,
        confirm_frames=2,
        drop_frames=3,
        ema_alpha=0.4,
        max_sigma=3.0,
    )
    index = FaceIndex(metric="ip", normalize=True)
    assigner = HysteresisAssigner(index, cfg)

    label, debug = assigner.assign(_vec(0.2, 0.1, 0.97), frame_idx=0, track_id="3")
    assert label.startswith("ANON_")
    assert assigner.state["3"].status == "anonymous"
    assert "reason" not in debug
