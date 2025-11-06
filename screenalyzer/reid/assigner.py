"""Hysteresis-based Re-ID assigner that sits atop a FAISS face index."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .faiss_index import FaceIndex, LabelStats


@dataclass
class ReIdConfig:
    """Configuration for hysteresis thresholds."""

    tau_join: float = 0.55
    tau_stay: float = 0.50
    tau_spawn: float = 0.45
    ema_window: int = 10
    confirm_frames: int = 5
    drop_frames: int = 8
    ema_alpha: float = 0.01
    max_sigma: float = 2.0


@dataclass
class TrackState:
    """Internal state per tracker ID."""

    label: str = "UNK"
    status: str = "anonymous"  # anonymous, provisional, confirmed
    ema: float = 0.0
    frames_above: int = 0
    frames_below: int = 0
    candidate: Optional[str] = None
    provisional_label: Optional[str] = None
    anonymous_label: Optional[str] = None
    last_cos: float = 0.0
    last_update_frame: int = -1
    last_confirmed_label: Optional[str] = None
    last_added_frame: Optional[int] = None
    confirmed_at: Optional[int] = None


class HysteresisAssigner:
    """Assign global identities to tracker IDs using hysteresis and FAISS search."""

    def __init__(self, index: FaceIndex, cfg: ReIdConfig, stats_store: Any = None):
        self.index = index
        self.cfg = cfg
        self.stats_store = stats_store
        self.state: Dict[str, TrackState] = {}
        self.id_map: Dict[str, str] = {}

        self._anon_counter = 0
        self._provisional_counter = 0

    # ------------------------------------------------------------------ helpers
    def _resolve_label(self, label: str) -> str:
        """Resolve label aliases created during merges."""
        if not label:
            return label
        resolved = label
        visited: set[str] = set()
        while resolved in self.id_map and resolved not in visited:
            visited.add(resolved)
            resolved = self.id_map[resolved]
        return resolved

    def _stats_for_label(self, label: Optional[str]) -> Optional[LabelStats]:
        if not label:
            return None
        resolved = self._resolve_label(label)
        stats = self.index.get_label_stats(resolved)
        if stats is None and resolved != label:
            stats = self.index.get_label_stats(label)
        return stats

    def _candidate_from_search(self, vector: np.ndarray) -> Optional[Dict[str, Any]]:
        results = self.index.search(vector, k=5)
        for raw_label, raw_score in results:
            resolved = self._resolve_label(raw_label)
            stats = self._stats_for_label(resolved) or self._stats_for_label(raw_label)
            if stats is None or stats.centroid.size == 0:
                continue
            cos = float(np.dot(vector, stats.centroid))
            dist = float(np.linalg.norm(vector - stats.centroid))
            sigma = stats.sigma
            z = dist / sigma if sigma > 1e-6 else None
            return {
                "label": resolved,
                "raw_label": raw_label,
                "raw_score": float(raw_score),
                "cos": cos,
                "dist": dist,
                "sigma": sigma,
                "z": z,
            }
        return None

    def _update_ema(self, prev: float, value: float) -> float:
        if prev == 0.0:
            return value
        return float(prev + self.cfg.ema_alpha * (value - prev))

    def _within_sigma(self, candidate: Optional[Dict[str, Any]]) -> bool:
        if not candidate:
            return False
        z = candidate.get("z")
        if z is None:
            return True
        return z <= self.cfg.max_sigma

    def _log_event(self, event: Dict[str, Any]) -> None:
        if self.stats_store is None:
            return
        if hasattr(self.stats_store, "append"):
            self.stats_store.append(event)
        elif hasattr(self.stats_store, "log"):
            self.stats_store.log(event)

    # ------------------------------------------------------------------ assignment flows
    def _assign_anonymous(
        self,
        state: TrackState,
        vector: np.ndarray,
        candidate: Optional[Dict[str, Any]],
        frame_idx: int,
    ) -> Tuple[str, Optional[str], str, Dict[str, Any]]:
        target_cos = candidate["cos"] if candidate else 0.0
        if candidate and candidate["cos"] >= self.cfg.tau_join and self._within_sigma(candidate):
            prev_label = state.label
            new_label = candidate["label"]
            state.status = "confirmed"
            state.label = new_label
            state.ema = target_cos
            state.frames_above = 1
            state.frames_below = 0
            state.candidate = None
            prior_confirmed = state.last_confirmed_label
            state.last_confirmed_label = new_label
            state.confirmed_at = frame_idx

            if state.provisional_label:
                self.id_map[state.provisional_label] = new_label
                state.provisional_label = None
            if state.anonymous_label:
                self.id_map[state.anonymous_label] = new_label

            self.index.update_centroid(new_label, vector, alpha=self.cfg.ema_alpha)

            reason = "reattach" if prior_confirmed == new_label else "join_confirmed"
            decision = "confirm_from_anonymous"
            return new_label, reason, decision, {"prev_label": prev_label, "cos": target_cos}

        if candidate and candidate["cos"] >= self.cfg.tau_spawn:
            if state.provisional_label is None:
                state.provisional_label = f"PROV_{self._provisional_counter:04d}"
                self._provisional_counter += 1
            state.status = "provisional"
            state.label = state.provisional_label
            state.candidate = candidate["label"]
            state.ema = candidate["cos"]
            state.frames_above = 1 if state.ema >= self.cfg.tau_join else 0
            state.frames_below = 0
            decision = "spawn_provisional"
            reason = "assign_provisional"
            return state.label, reason, decision, {"candidate": candidate["label"], "cos": candidate["cos"]}

        if state.anonymous_label is None:
            state.anonymous_label = f"ANON_{self._anon_counter:04d}"
            self._anon_counter += 1
            self.index.add(state.anonymous_label, vector)

        state.status = "anonymous"
        state.label = state.anonymous_label
        state.ema = target_cos
        state.candidate = None
        self.index.update_centroid(state.anonymous_label, vector, alpha=self.cfg.ema_alpha)

        return state.label, None, "stay_anonymous", {"cos": target_cos}

    def _assign_confirmed(
        self,
        state: TrackState,
        vector: np.ndarray,
        candidate: Optional[Dict[str, Any]],
        _frame_idx: int,
    ) -> Tuple[str, Optional[str], str, Dict[str, Any]]:
        stats = self._stats_for_label(state.label)
        cos_current = float(np.dot(vector, stats.centroid)) if stats else 0.0
        state.last_cos = cos_current
        state.ema = self._update_ema(state.ema, cos_current)

        if state.ema >= self.cfg.tau_stay:
            state.frames_above += 1
            state.frames_below = 0
        else:
            state.frames_below += 1
            state.frames_above = 0

        if state.frames_below >= self.cfg.drop_frames:
            prev_label = state.label
            state.status = "anonymous"
            state.label = "UNK"
            state.ema = 0.0
            state.frames_above = 0
            state.frames_below = 0
            state.candidate = None
            state.last_confirmed_label = prev_label
            state.anonymous_label = prev_label if prev_label.startswith("ANON_") else state.anonymous_label
            return "UNK", "drop_exit", "drop_confirmed", {"prev_label": prev_label, "cos": cos_current}

        if candidate and candidate["label"] != state.label:
            if candidate["cos"] >= self.cfg.tau_join and self._within_sigma(candidate):
                prev_label = state.label
                state.label = candidate["label"]
                state.status = "confirmed"
                state.ema = candidate["cos"]
                state.frames_above = 1
                state.frames_below = 0
                state.last_confirmed_label = state.label
                self.index.update_centroid(state.label, vector, alpha=self.cfg.ema_alpha)
                return state.label, "switch", "switch_confirmed", {"prev_label": prev_label, "cos": candidate["cos"]}

        if stats:
            self.index.update_centroid(state.label, vector, alpha=self.cfg.ema_alpha)

        return state.label, None, "stay_confirmed", {"cos": cos_current, "ema": state.ema}

    def _assign_provisional(
        self,
        state: TrackState,
        vector: np.ndarray,
        candidate: Optional[Dict[str, Any]],
        frame_idx: int,
    ) -> Tuple[str, Optional[str], str, Dict[str, Any]]:
        if state.candidate is None:
            state.status = "anonymous"
            state.ema = 0.0
            return self._assign_anonymous(state, vector, candidate, frame_idx)

        stats = self._stats_for_label(state.candidate)
        if stats is None or stats.centroid.size == 0:
            state.status = "anonymous"
            state.ema = 0.0
            state.candidate = None
            return self._assign_anonymous(state, vector, candidate, frame_idx)

        cos = float(np.dot(vector, stats.centroid))
        dist = float(np.linalg.norm(vector - stats.centroid))
        sigma = stats.sigma
        z = dist / sigma if sigma > 1e-6 else None

        state.last_cos = cos
        state.ema = self._update_ema(state.ema, cos) if state.ema else cos
        if state.ema >= self.cfg.tau_join:
            state.frames_above += 1
        else:
            state.frames_above = 0

        if cos < self.cfg.tau_spawn:
            state.status = "anonymous"
            state.candidate = None
            state.ema = 0.0
            state.frames_above = 0
            return self._assign_anonymous(state, vector, candidate, frame_idx)

        if (
            state.frames_above >= self.cfg.confirm_frames
            and state.ema >= self.cfg.tau_join
            and (z is None or z <= self.cfg.max_sigma)
        ):
            final_label = state.candidate
            prev_label = state.label
            state.status = "confirmed"
            state.label = final_label
            state.candidate = None
            state.frames_above = 0
            state.frames_below = 0
            state.ema = cos
            state.last_confirmed_label = final_label
            state.confirmed_at = frame_idx

            if state.provisional_label:
                self.id_map[state.provisional_label] = final_label
                state.provisional_label = None
            if state.anonymous_label:
                self.id_map[state.anonymous_label] = final_label

            self.index.update_centroid(final_label, vector, alpha=self.cfg.ema_alpha)
            return final_label, "promote_confirm", "confirm_provisional", {
                "prev_label": prev_label,
                "cos": cos,
                "z": z,
            }

        if candidate and candidate["label"] != state.candidate and candidate["cos"] >= self.cfg.tau_spawn + 0.02:
            state.candidate = candidate["label"]
            state.frames_above = 0
            state.ema = candidate["cos"]

        return state.label, None, "stay_provisional", {"cos": cos, "ema": state.ema, "z": z}

    # ------------------------------------------------------------------ public API
    def assign(self, emb: np.ndarray, frame_idx: int, track_id: Optional[str]) -> Tuple[str, Dict[str, Any]]:
        vector = np.asarray(emb, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector = vector / norm

        key = str(track_id) if track_id is not None else f"__tmp_{frame_idx}"
        state = self.state.get(key)
        if state is None:
            state = TrackState()
            self.state[key] = state

        prev_label = state.label
        prev_status = state.status

        candidate = self._candidate_from_search(vector)

        if state.status == "confirmed":
            new_label, reason, decision, extra = self._assign_confirmed(state, vector, candidate, frame_idx)
        elif state.status == "provisional":
            new_label, reason, decision, extra = self._assign_provisional(state, vector, candidate, frame_idx)
        else:
            new_label, reason, decision, extra = self._assign_anonymous(state, vector, candidate, frame_idx)

        state.last_update_frame = frame_idx

        debug = {
            "frame": frame_idx,
            "track_id": track_id,
            "label_before": prev_label,
            "label_after": new_label,
            "status_before": prev_status,
            "status_after": state.status,
            "ema": state.ema,
            "decision": decision,
            "candidate_label": candidate["label"] if candidate else None,
            "candidate_cos": candidate["cos"] if candidate else None,
            "candidate_z": candidate["z"] if candidate else None,
        }
        debug.update(extra or {})

        if reason:
            debug["reason"] = reason
            self._log_event(debug)

        if track_id is None:
            self.state.pop(key, None)

        return new_label, debug

    def end_track(self, track_id: str) -> None:
        self.state.pop(str(track_id), None)
