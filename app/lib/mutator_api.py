"""
Workspace mutator API.

Provides a stable, episode-scoped interface for cluster/track mutations
with automatic analytics dirty marking, diagnostics logging, and
post-action cache refresh so UI state updates immediately.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import yaml

from app.lib.cluster_mutations import ClusterMutator
from app.lib.data import (
    load_cluster_metrics,
    load_clusters,
    load_person_metrics,
    load_tracks,
)


def _load_confidence_thresholds(config_path: Path) -> Dict[str, float]:
    """Load confidence thresholds from pipeline config."""
    if not config_path.exists():
        return {}

    try:
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}
        return config.get("confidence", {}) or {}
    except Exception:
        return {}


def _safe_dataframe(records: list[dict], columns: list[str]) -> pd.DataFrame:
    """Return DataFrame from records with default schema."""
    if not records:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(records)


@dataclass
class WorkspaceState:
    """Cached workspace data for current episode."""

    clusters_data: dict = field(default_factory=dict)
    tracks_data: dict = field(default_factory=dict)
    cluster_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    person_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    track_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)


class WorkspaceMutator:
    """Episode-scoped mutation context with cached data access."""

    def __init__(self, episode_id: str, data_root: Path = Path("data")):
        self.episode_id = episode_id
        self.data_root = Path(data_root)
        self._thresholds = _load_confidence_thresholds(Path("configs/pipeline.yaml"))
        self.state = WorkspaceState()
        self.refresh()

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    @property
    def thresholds(self) -> Dict[str, float]:
        return dict(self._thresholds)

    @property
    def clusters(self) -> dict:
        return self.state.clusters_data

    @property
    def tracks(self) -> dict:
        return self.state.tracks_data

    @property
    def cluster_metrics_df(self) -> pd.DataFrame:
        return self.state.cluster_metrics.copy()

    @property
    def person_metrics_df(self) -> pd.DataFrame:
        return self.state.person_metrics.copy()

    @property
    def track_metrics_df(self) -> pd.DataFrame:
        return self.state.track_metrics.copy()

    def get_low_confidence_clusters(self) -> pd.DataFrame:
        df = self.cluster_metrics_df
        if df.empty:
            return df

        low_threshold = float(self._thresholds.get("cluster_low_p25", 0.6))
        contam_threshold = float(self._thresholds.get("cluster_contam_high", 0.2))

        mask = (df["tracks_conf_p25_median"] < low_threshold) | (
            df["contam_rate"] >= contam_threshold
        )
        return df[mask].copy()

    def get_low_confidence_tracks(self) -> pd.DataFrame:
        df = self.track_metrics_df
        if df.empty:
            return df

        track_low = float(self._thresholds.get("track_low_p25", 0.55))
        conflict_high = float(self._thresholds.get("track_conflict_frac_high", 0.2))

        mask = (df["conf_p25"] < track_low) | (df["conflict_frac"] >= conflict_high)
        return df[mask].copy()

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        """Reload clusters/tracks + metrics from disk."""
        clusters_data = load_clusters(self.episode_id, self.data_root) or {"clusters": []}
        tracks_data = load_tracks(self.episode_id, self.data_root) or {"tracks": []}

        cluster_metrics_df = load_cluster_metrics(self.episode_id, self.data_root)
        if cluster_metrics_df is None:
            cluster_metrics_df = self._build_cluster_metrics_fallback(clusters_data)

        person_metrics_df = load_person_metrics(self.episode_id, self.data_root)
        if person_metrics_df is None:
            person_metrics_df = self._build_person_metrics_fallback(clusters_data)

        track_metrics_df = self._build_track_metrics(clusters_data, tracks_data)

        self.state = WorkspaceState(
            clusters_data=clusters_data,
            tracks_data=tracks_data,
            cluster_metrics=cluster_metrics_df,
            person_metrics=person_metrics_df,
            track_metrics=track_metrics_df,
        )

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    def assign_name(self, cluster_id: int, identity: str, lock: bool = True) -> dict:
        payload = {"cluster_id": cluster_id, "identity": identity, "lock": lock}
        return self._perform_action("assign_name", payload, lambda mut: mut.assign_name(cluster_id, identity, lock))

    def assign_tracks_to_identity(
        self,
        track_ids: list[int],
        identity: str,
        source_cluster_id: Optional[int] = None,
    ) -> dict:
        payload = {
            "track_ids": track_ids,
            "identity": identity,
            "source_cluster_id": source_cluster_id,
        }
        return self._perform_action(
            "assign_tracks_to_identity",
            payload,
            lambda mut: mut.assign_tracks_to_identity(track_ids, identity, source_cluster_id),
        )

    def split_frames_and_assign(
        self,
        track_id: int,
        frame_ids: list[int],
        identity: str,
    ) -> dict:
        payload = {"track_id": track_id, "frame_ids": frame_ids, "identity": identity}
        return self._perform_action(
            "split_frames_and_assign",
            payload,
            lambda mut: mut.split_frames_and_assign(track_id, frame_ids, identity),
        )

    def move_track(self, track_id: int, from_cluster_id: int, to_cluster_id: int) -> dict:
        payload = {
            "track_id": track_id,
            "from_cluster_id": from_cluster_id,
            "to_cluster_id": to_cluster_id,
        }
        return self._perform_action(
            "move_track",
            payload,
            lambda mut: mut.move_track(track_id, from_cluster_id, to_cluster_id),
        )

    def delete_track(self, track_id: int, cluster_id: int) -> dict:
        payload = {"track_id": track_id, "cluster_id": cluster_id}
        return self._perform_action(
            "delete_track",
            payload,
            lambda mut: mut.delete_track(track_id, cluster_id),
        )

    def delete_cluster(self, cluster_id: int) -> dict:
        payload = {"cluster_id": cluster_id}

        def executor(mut: ClusterMutator):
            clusters = load_clusters(self.episode_id, self.data_root) or {"clusters": []}
            target = next(
                (c for c in clusters.get("clusters", []) if c.get("cluster_id") == cluster_id),
                None,
            )
            if not target:
                raise ValueError(f"Cluster {cluster_id} not found")

            for track_id in list(target.get("track_ids", [])):
                mut.delete_track(track_id, cluster_id)

            mut._mark_analytics_dirty("cluster deleted")  # type: ignore[attr-defined]
            return {"deleted_cluster_id": cluster_id}

        return self._perform_action("delete_cluster", payload, executor)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _perform_action(self, action: str, payload: dict, executor) -> dict:
        mutator = ClusterMutator(self.episode_id, self.data_root)
        ok = True
        error_msg = None
        result: Dict[str, Any]

        try:
            result = executor(mutator)
            self.refresh()
        except Exception as exc:
            ok = False
            error_msg = str(exc)
            result = {}

        metrics_snapshot = self._collect_metrics_snapshot(payload)
        self._log_ui_action(action, payload, metrics_snapshot, ok, error_msg)

        return {
            "ok": ok,
            "result": result,
            "error": error_msg,
            "metrics": metrics_snapshot,
        }

    def _collect_metrics_snapshot(self, payload: dict) -> dict:
        """Collect relevant metrics for logging purposes."""
        cluster_id = payload.get("cluster_id") or payload.get("source_cluster_id")
        track_ids = payload.get("track_ids") or (
            [payload["track_id"]] if "track_id" in payload else []
        )

        cluster_metrics = None
        if cluster_id is not None:
            df = self.cluster_metrics_df
            if not df.empty and int(cluster_id) in df["cluster_id"].values:
                cluster_metrics = (
                    df[df["cluster_id"] == int(cluster_id)].iloc[0].to_dict()
                )

        track_metrics = []
        if track_ids:
            df = self.track_metrics_df
            if not df.empty:
                subset = df[df["track_id"].isin([int(tid) for tid in track_ids])]
                if not subset.empty:
                    track_metrics = subset.to_dict(orient="records")

        return {
            "cluster": cluster_metrics,
            "tracks": track_metrics,
        }

    def _log_ui_action(
        self,
        action: str,
        payload: dict,
        metrics: dict,
        ok: bool,
        error: Optional[str],
    ) -> None:
        diagnostics_dir = (
            self.data_root / "harvest" / self.episode_id / "diagnostics"
        )
        diagnostics_dir.mkdir(parents=True, exist_ok=True)

        log_path = diagnostics_dir / "ui_workspace_log.jsonl"
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "action": action,
            "payload": payload,
            "metrics": metrics,
            "ok": ok,
            "error": error,
        }

        with open(log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    # ------------------------------------------------------------------
    # Metric builders
    # ------------------------------------------------------------------

    def _build_cluster_metrics_fallback(self, clusters_data: dict) -> pd.DataFrame:
        records: list[dict] = []
        for cluster in clusters_data.get("clusters", []):
            track_metrics = cluster.get("track_metrics", []) or []
            conf_values = [tm.get("conf_p25", 0.0) for tm in track_metrics]
            median_p25 = float(np.median(conf_values)) if conf_values else 0.0
            min_p25 = float(np.min(conf_values)) if conf_values else 0.0
            records.append(
                {
                    "cluster_id": int(cluster.get("cluster_id")),
                    "name": cluster.get("name", "Unknown"),
                    "n_tracks": len(cluster.get("track_ids", [])),
                    "tracks_conf_p25_median": median_p25,
                    "contam_rate": 0.0,
                    "pairwise_centroid_sim_mean": float(
                        cluster.get("quality_score", 0.0)
                    ),
                    "min_track_conf_p25": min_p25,
                }
            )

        return _safe_dataframe(
            records,
            [
                "cluster_id",
                "name",
                "n_tracks",
                "tracks_conf_p25_median",
                "contam_rate",
                "pairwise_centroid_sim_mean",
                "min_track_conf_p25",
            ],
        )

    def _build_person_metrics_fallback(self, clusters_data: dict) -> pd.DataFrame:
        identity_to_tracks: dict[str, list[float]] = {}

        for cluster in clusters_data.get("clusters", []):
            name = cluster.get("name", "Unknown")
            if name == "Unknown":
                continue
            conf_values = [
                tm.get("conf_p25", 0.0) for tm in cluster.get("track_metrics", []) or []
            ]
            identity_to_tracks.setdefault(name, []).extend(conf_values)

        records: list[dict] = []
        for identity, conf_values in identity_to_tracks.items():
            median_conf = float(np.median(conf_values)) if conf_values else 0.0
            records.append(
                {
                    "person": identity,
                    "n_clusters": sum(
                        1
                        for cluster in clusters_data.get("clusters", [])
                        if cluster.get("name") == identity
                    ),
                    "n_tracks": len(conf_values),
                    "bank_conf_median_p25": median_conf,
                    "bank_contam_rate": 0.0,
                    "inter_id_margin": 0.0,
                }
            )

        return _safe_dataframe(
            records,
            [
                "person",
                "n_clusters",
                "n_tracks",
                "bank_conf_median_p25",
                "bank_contam_rate",
                "inter_id_margin",
            ],
        )

    def _build_track_metrics(self, clusters_data: dict, tracks_data: dict) -> pd.DataFrame:
        track_records: list[dict] = []

        track_lookup = {
            track["track_id"]: track for track in tracks_data.get("tracks", [])
        }

        for cluster in clusters_data.get("clusters", []):
            cluster_id = int(cluster.get("cluster_id"))
            person_name = cluster.get("name", "Unknown")
            for tm in cluster.get("track_metrics", []) or []:
                track_id = int(tm.get("track_id"))
                track_info = track_lookup.get(track_id, {})
                track_records.append(
                    {
                        "track_id": track_id,
                        "cluster_id": cluster_id,
                        "person": person_name,
                        "conf_p25": float(tm.get("conf_p25", 0.0)),
                        "conf_mean": float(tm.get("conf_mean", 0.0)),
                        "conf_min": float(tm.get("conf_min", 0.0)),
                        "conflict_frac": float(tm.get("conflict_frac", 0.0)),
                        "intra_var": float(tm.get("intra_var", 0.0)),
                        "n_low": int(tm.get("n_low", 0)),
                        "n_frames": int(tm.get("n_frames", 0)),
                        "avg_margin": float(tm.get("avg_margin", 0.0)),
                        "margin_p25": float(tm.get("margin_p25", 0.0)),
                        "start_ms": track_info.get("start_ms"),
                        "end_ms": track_info.get("end_ms"),
                    }
                )

        return _safe_dataframe(
            track_records,
            [
                "track_id",
                "cluster_id",
                "person",
                "conf_p25",
                "conf_mean",
                "conf_min",
                "conflict_frac",
                "intra_var",
                "n_low",
                "n_frames",
                "avg_margin",
                "margin_p25",
                "start_ms",
                "end_ms",
            ],
        )


# ------------------------------------------------------------------------------
# Module-level singleton helpers
# ------------------------------------------------------------------------------

_CONTEXT: Optional[WorkspaceMutator] = None


def configure_workspace_mutator(
    episode_id: str,
    data_root: Path = Path("data"),
) -> WorkspaceMutator:
    """Configure global workspace mutator for episode."""
    global _CONTEXT
    _CONTEXT = WorkspaceMutator(episode_id, data_root)
    return _CONTEXT


def get_workspace_mutator() -> WorkspaceMutator:
    if _CONTEXT is None:
        raise RuntimeError("Workspace mutator not configured. Call configure_workspace_mutator first.")
    return _CONTEXT


def assign_name(cluster_id: int, identity: str, lock: bool = True) -> dict:
    return get_workspace_mutator().assign_name(cluster_id, identity, lock)


def assign_tracks_to_identity(
    track_ids: list[int],
    identity: str,
    source_cluster_id: Optional[int] = None,
) -> dict:
    return get_workspace_mutator().assign_tracks_to_identity(track_ids, identity, source_cluster_id)


def split_frames_and_assign(
    track_id: int,
    frame_ids: list[int],
    identity: str,
) -> dict:
    return get_workspace_mutator().split_frames_and_assign(track_id, frame_ids, identity)


def move_track(track_id: int, from_cluster_id: int, to_cluster_id: int) -> dict:
    return get_workspace_mutator().move_track(track_id, from_cluster_id, to_cluster_id)


def delete_track(track_id: int, cluster_id: int) -> dict:
    return get_workspace_mutator().delete_track(track_id, cluster_id)


def delete_cluster(cluster_id: int) -> dict:
    return get_workspace_mutator().delete_cluster(cluster_id)
