"""Reference Poker44 miner with simple chunk-level behavioral heuristics."""

import hashlib
import json
import os
import subprocess
import time
import atexit
import importlib
import queue
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Tuple

import bittensor as bt
import numpy as np
import pandas as pd

from poker44.base.miner import BaseMinerNeuron
from poker44.validator.chunk_features import aggregate_chunk_from_hands
from poker44.utils.model_manifest import (
    build_local_model_manifest,
    evaluate_manifest_compliance,
    manifest_digest,
)
from poker44.validator.synapse import DetectionSynapse


class ScoreMonitor:
    """Rolling window monitor that detects bot profile rotation from unlabeled score distributions.

    How it works (dummy example):
        Monday — model scores arrive:
          [0.02, 0.88, 0.01, 0.91, 0.03, 0.85]
          Two clear clusters: humans near 0.02, bots near 0.88
          → profile_signature = {"human_peak": 0.02, "bot_peak": 0.88}

        Tuesday — new scores arrive:
          [0.02, 0.04, 0.01, 0.05, 0.03, 0.06]
          Still bimodal but gap is tiny: humans ~0.02, bots ~0.05
          → distribution shifted! Bot peak moved from 0.88 → 0.05
          → ALERT: "Rotation detected — bots became passive"

    The monitor tracks:
      1. Rolling score buffer (last N scores)
      2. Periodically computes bimodal split via largest-gap heuristic
      3. Compares current profile signature against previous
      4. Logs alert when shift exceeds threshold
    """

    def __init__(self, window: int = 200, shift_threshold: float = 0.15):
        self._window = window
        self._shift_threshold = shift_threshold
        self._scores: deque = deque(maxlen=window)
        self._prev_signature: dict | None = None
        self._request_count = 0
        self._check_every = 5  # analyze every N requests
        self._last_alert: str = ""

    def add_scores(self, scores: list[float]) -> None:
        self._scores.extend(scores)
        self._request_count += 1
        if self._request_count % self._check_every == 0 and len(self._scores) >= 20:
            self._analyze()

    def _find_largest_gap(self, arr: np.ndarray) -> tuple[float, int]:
        sorted_s = np.sort(arr)
        gaps = np.diff(sorted_s)
        if len(gaps) == 0:
            return 0.0, -1
        idx = int(np.argmax(gaps))
        return float(gaps[idx]), idx

    def _analyze(self) -> None:
        arr = np.array(list(self._scores))
        sorted_s = np.sort(arr)
        max_gap, gap_idx = self._find_largest_gap(arr)
        median_gap = float(np.median(np.diff(sorted_s))) if len(sorted_s) > 1 else 0.0

        is_bimodal = max_gap > 5 * max(median_gap, 1e-6) and max_gap > 0.01
        if is_bimodal and gap_idx >= 0:
            boundary = (sorted_s[gap_idx] + sorted_s[gap_idx + 1]) / 2
            low_cluster = arr[arr < boundary]
            high_cluster = arr[arr >= boundary]
            sig = {
                "human_peak": round(float(np.median(low_cluster)), 4),
                "bot_peak": round(float(np.median(high_cluster)), 4),
                "gap": round(float(max_gap), 4),
                "n_low": len(low_cluster),
                "n_high": len(high_cluster),
                "bimodal": True,
            }
        else:
            sig = {
                "human_peak": round(float(np.median(arr)), 4),
                "bot_peak": None,
                "gap": round(float(max_gap), 4),
                "n_low": len(arr),
                "n_high": 0,
                "bimodal": False,
            }

        if self._prev_signature is not None:
            prev_bot = self._prev_signature.get("bot_peak")
            curr_bot = sig.get("bot_peak")
            if prev_bot is not None and curr_bot is not None:
                shift = abs(curr_bot - prev_bot)
                if shift > self._shift_threshold:
                    self._last_alert = (
                        f"ROTATION DETECTED: bot peak shifted {prev_bot:.3f} → {curr_bot:.3f} "
                        f"(delta={shift:.3f})"
                    )
                    bt.logging.warning(f"[ScoreMonitor] {self._last_alert}")
            elif prev_bot is not None and curr_bot is None:
                self._last_alert = "ALERT: Lost bimodal structure — bots may be very passive"
                bt.logging.warning(f"[ScoreMonitor] {self._last_alert}")
            elif prev_bot is None and curr_bot is not None:
                self._last_alert = f"Bimodal detected: bot_peak={curr_bot:.3f}"
                bt.logging.info(f"[ScoreMonitor] {self._last_alert}")

        bt.logging.info(
            f"[ScoreMonitor] n={len(arr)} bimodal={sig['bimodal']} "
            f"human_peak={sig['human_peak']} bot_peak={sig['bot_peak']} gap={sig['gap']} "
            f"low={sig['n_low']} high={sig['n_high']}"
        )
        self._prev_signature = sig

    @property
    def last_alert(self) -> str:
        return self._last_alert

    @property
    def current_signature(self) -> dict | None:
        return self._prev_signature


REPO_ROOT = Path(__file__).resolve().parents[1]


def _hash_feature_name(name: str, salt: str) -> str:
    """Deterministic hash of a feature name with a secret salt."""
    raw = hashlib.sha256((salt + name).encode("utf-8")).hexdigest()[:10]
    return f"f_{raw}"


def _build_feature_hash_mapping(raw_names: list[str], salt: str) -> dict[str, str]:
    """Build raw_name → hashed_name mapping for all known features."""
    return {name: _hash_feature_name(name, salt) for name in raw_names}


def _compute_invariant_features(row: dict[str, float]) -> dict[str, float]:
    """Compute domain-invariant ratio features from raw aggregate features.

    Replaces absolute magnitude features (pot, bet_size, norm_bb) with
    self-referencing ratios that capture behavioral PATTERNS without
    depending on blind structure, table size, or era.
    """
    out = dict(row)
    eps = 1e-8

    pot_base = max(float(out.get("mean_pot_after_mean", 0.0)), eps)

    if "mean_pot_after_std" in out:
        out["pot_cv_inv"] = out["mean_pot_after_std"] / pot_base
    if "mean_pot_after_p90" in out and "mean_pot_after_p10" in out:
        out["pot_spread_inv"] = (out["mean_pot_after_p90"] - out["mean_pot_after_p10"]) / pot_base
    if "mean_pot_after_p50" in out and "mean_pot_after_p90" in out and "mean_pot_after_p10" in out:
        pot_range = max(out["mean_pot_after_p90"] - out["mean_pot_after_p10"], eps)
        out["pot_median_position_inv"] = (out["mean_pot_after_p50"] - out["mean_pot_after_p10"]) / pot_range
    if "pot_growth_mean" in out:
        out["pot_growth_relative_inv"] = out["pot_growth_mean"] / pot_base
    if "pot_growth_std" in out:
        out["pot_growth_cv_inv"] = out["pot_growth_std"] / pot_base
    if "std_pot_after_std" in out:
        out["std_pot_cv_inv"] = out["std_pot_after_std"] / pot_base

    bet_base = max(float(out.get("bet_size_mean_mean", 0.0)), eps)
    if "bet_size_mean_std" in out:
        out["bet_size_cv_inv"] = out["bet_size_mean_std"] / bet_base
    if "bet_size_mean_p90" in out:
        out["bet_size_p90_ratio_inv"] = out["bet_size_mean_p90"] / bet_base
    if "bet_size_mean_p50" in out:
        out["bet_size_p50_ratio_inv"] = out["bet_size_mean_p50"] / bet_base
    if "bet_size_mean_max" in out:
        out["bet_size_max_ratio_inv"] = out["bet_size_mean_max"] / bet_base
    if "bet_size_max_std" in out:
        out["bet_size_max_cv_inv"] = out["bet_size_max_std"] / bet_base
    if "bet_size_std_max" in out:
        out["bet_size_std_max_ratio_inv"] = out["bet_size_std_max"] / bet_base

    if "mean_norm_bb_p90" in out and "std_norm_bb_std" in out:
        bb_base = max(float(out["mean_norm_bb_p90"]), eps)
        out["norm_bb_cv_inv"] = out["std_norm_bb_std"] / bb_base
    if "max_norm_bb_std" in out and "mean_norm_bb_p90" in out:
        bb_base = max(float(out["mean_norm_bb_p90"]), eps)
        out["max_norm_bb_cv_inv"] = out["max_norm_bb_std"] / bb_base

    if "bet_size_mean_mean" in out and "mean_pot_after_mean" in out:
        out["bet_to_pot_ratio_inv"] = float(out["bet_size_mean_mean"]) / pot_base
    if "pot_growth_mean" in out and "pot_growth_std" in out:
        pg_mean = max(float(out["pot_growth_mean"]), eps)
        out["pot_growth_consistency_inv"] = out["pot_growth_std"] / pg_mean

    for k in list(out.keys()):
        if k.endswith("_inv"):
            v = out[k]
            if not np.isfinite(v):
                out[k] = 0.0

    return out


def _git_head_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def _parse_rotate_seconds(spec: str) -> Optional[float]:
    """
    Parse a rotation interval for miner request logs.

    Supported:
    - "0" / "false" / "off" -> None (no rotation)
    - "1" / "true" / "yes" / "on" -> 24h (backward compatible)
    - "Ns" / "Nm" / "Nh" / "Nd" -> seconds
    - plain number -> seconds
    """
    s = (spec or "").strip().lower()
    if not s or s in {"0", "false", "off", "no", "none"}:
        return None
    if s in {"1", "true", "yes", "on"}:
        return 24.0 * 3600.0
    # Duration suffixes
    for suf, mult in (("s", 1.0), ("m", 60.0), ("h", 3600.0), ("d", 86400.0)):
        if s.endswith(suf):
            return float(s[: -len(suf)].strip()) * mult
    # Fallback: interpret as seconds
    return float(s)

class Miner(BaseMinerNeuron):
    """
    Reference heuristic miner.

    Scoring uses the same **chunk schema** as ``preprocess_lightgbm`` / LGBM training:
    ``sanitize_hand_for_miner`` → per-hand numeric features → mean/std/max over the
    chunk → optional ``transform_meta`` → optional ``dbf_*`` (train-fitted bounds JSON)
    → optional SSL ``emb_*`` → one score in ``[0, 1]``. The validator still receives
    **one risk score per chunk** (not per hand).

    The goal is not SOTA accuracy, but a deterministic and explainable baseline
    that is meaningfully better than random and aligned with miner-visible features.
    """

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)
        bt.logging.info("🤖 Heuristic Poker44 Miner started")
        self._request_logging_enabled = os.getenv("POKER44_MINER_LOG_REQUESTS", "0").strip() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self._request_logging_full_chunks = os.getenv(
            "POKER44_MINER_LOG_FULL_CHUNKS", "0"
        ).strip() in {"1", "true", "yes", "on"}
        # If true: each log line is only {"chunks": [...]} when chunks are logged — no
        # ts_utc, request_idx, chunk_hashes, risk_scores, validator_hotkey, etc.
        # Skips logging entirely when chunks are not attached (high/extreme adaptive,
        # full_chunks off, or empty request). Saves file size and avoids chunk_hash CPU.
        # Note: workspace/real_distribution/process.py in-place --unique-by chunk needs
        # chunk_hashes; use --merge or --unique-by hand with slim lines.
        self._request_log_ssl_export = os.getenv(
            "POKER44_MINER_LOG_SSL_EXPORT", "0"
        ).strip() in {"1", "true", "yes", "on"}
        # One JSON line per chunk: {"chunk_hash","chunk","risk_score"} (raw score if available).
        # Takes precedence over ssl/slim batched rows when enabled. Same gating as slim/ssl
        # for full_chunks + adaptive state.
        self._request_log_chunk_ndjson = os.getenv(
            "POKER44_MINER_LOG_CHUNK_NDJSON", "0"
        ).strip() in {"1", "true", "yes", "on"}
        self._request_log_slim_export = os.getenv(
            "POKER44_MINER_LOG_SLIM_EXPORT", "0"
        ).strip() in {"1", "true", "yes", "on"}
        log_path_env = os.getenv(
            "POKER44_MINER_LOG_PATH",
            str(REPO_ROOT / "workspace" / "real_distribution" / "miner_requests.jsonl"),
        )
        self._request_log_path = Path(log_path_env).expanduser().resolve()
        rotate_spec = os.getenv("POKER44_MINER_LOG_DAILY_ROTATE", "1h").strip()
        self._request_log_rotate_seconds = _parse_rotate_seconds(rotate_spec)
        # Anchor for interval-based rotation so test values like "5s" create
        # predictable sequential filenames while the miner runs.
        self._log_rotate_anchor = time.monotonic()
        self._log_async = os.getenv("POKER44_MINER_LOG_ASYNC", "0").strip() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self._log_queue_max = int(os.getenv("POKER44_MINER_LOG_QUEUE_MAX", "5000"))
        self._log_batch_size = int(os.getenv("POKER44_MINER_LOG_BATCH_SIZE", "64"))
        self._log_flush_ms = int(os.getenv("POKER44_MINER_LOG_FLUSH_MS", "1000"))
        self._log_near_full_ratio = float(
            os.getenv("POKER44_MINER_LOG_NEAR_FULL_RATIO", "0.80")
        )
        self._adaptive_logging = os.getenv(
            "POKER44_MINER_ADAPTIVE_LOGGING", "0"
        ).strip() in {"1", "true", "yes", "on"}
        self._timeout_budget_ms = float(
            os.getenv("POKER44_MINER_VALIDATOR_TIMEOUT_MS", "20000")
        )
        self._high_load_ratio = float(os.getenv("POKER44_MINER_HIGH_LOAD_RATIO", "0.50"))
        self._extreme_load_ratio = float(
            os.getenv("POKER44_MINER_EXTREME_LOAD_RATIO", "0.80")
        )
        self._latency_window = int(os.getenv("POKER44_MINER_LATENCY_WINDOW", "200"))
        self._latency_hist: deque[float] = deque(maxlen=max(20, self._latency_window))
        self._log_state = "normal"
        self._dropped_log_rows = 0
        self._pending_flush_signal = False
        self._log_queue: Optional[queue.Queue] = None
        self._log_worker: Optional[threading.Thread] = None
        self._log_worker_stop = threading.Event()
        self._other_only_mode = os.getenv(
            "POKER44_MINER_OTHER_ONLY", "0"
        ).strip() in {"1", "true", "yes", "on"}
        if self._other_only_mode:
            bt.logging.info(
                "🎯 OTHER-ONLY mode enabled: scoring purely on other_ratio_mean signal."
            )
        self._uncertain_a = float(os.getenv("POKER44_MINER_UNCERTAIN_A", "-1"))
        self._uncertain_b = float(os.getenv("POKER44_MINER_UNCERTAIN_B", "-1"))
        self._uncertain_gamma = float(os.getenv("POKER44_MINER_UNCERTAIN_GAMMA", "1.0"))
        _ua, _ub, _ug = self._uncertain_a, self._uncertain_b, self._uncertain_gamma
        if 0.0 <= _ua < _ub <= 1.0 and _ug > 0.0:
            bt.logging.info(
                f"Uncertain-band smoothing ON: POKER44_MINER_UNCERTAIN_A/B/GAMMA={_ua}/{_ub}/{_ug}"
            )
        else:
            bt.logging.info(
                "Uncertain-band smoothing OFF (need 0<=a<b<=1 and gamma>0); "
                f"POKER44_MINER_UNCERTAIN_A/B/GAMMA={_ua}/{_ub}/{_ug}"
            )
        self._model_path = os.getenv(
            "POKER44_MINER_MODEL_PATH",
            str(
                REPO_ROOT
                / "workspace"
                / "model"
                / "artifacts"
                / "lgbm_2_v1"
                / "lgbm_classifier.joblib"
            ),
        ).strip()
        self._model_bundle_dir = os.getenv("POKER44_MINER_MODEL_BUNDLE_DIR", "").strip()
        self._transform_meta_path = os.getenv("POKER44_MINER_TRANSFORM_META_PATH", "").strip()
        self._wgz_meta_path = os.getenv("POKER44_MINER_WGZ_META_PATH", "").strip()
        # If true: refuse to start without a loaded model, and never fall back to heuristics
        # on a chunk (inference errors propagate; fix model or restart).
        self._miner_require_model = os.getenv(
            "POKER44_MINER_REQUIRE_MODEL", "0"
        ).strip() in {"1", "true", "yes", "on"}
        self._model: Optional[Any] = None
        self._model_features: Optional[list[str]] = None
        self._inference_threshold: float = 0.5
        self._score_monitor = ScoreMonitor(window=200, shift_threshold=0.15)
        self._feature_salt = os.getenv("POKER44_FEATURE_SALT", "").strip()
        self._feature_hash_map: Optional[dict[str, str]] = None
        self._score_remap_path = os.getenv("POKER44_SCORE_REMAP_PATH", "").strip()
        self._score_remap_anchors: Optional[list[tuple[float, float]]] = None
        self._maybe_load_score_remap()
        self._transform_meta: Optional[dict[str, Any]] = None
        self._wgz_meta: Optional[dict[str, Any]] = None
        # Optional ``ssl_masked_ae.npz`` — when set, ``emb_*`` columns are filled before LGBM predict.
        self._ssl_npz_path = os.getenv("POKER44_MINER_SSL_NPZ_PATH", "").strip()
        self._ssl_art: Optional[dict[str, Any]] = None
        # Train-fitted DBF winsor bounds (same JSON as ``prepare_ssl_embed_dbf_inputs`` /
        # ``dbf_quantile_bounds.json`` next to the joblib, or ``POKER44_MINER_DBF_BOUNDS_JSON``).
        self._dbf_bounds_path_env = os.getenv("POKER44_MINER_DBF_BOUNDS_JSON", "").strip()
        self._dbf_quantile_bounds: Optional[dict[str, tuple[float, float]]] = None
        self._tbm_adapter: Optional[dict[str, Any]] = None
        self._tbm_adapter_encoder: Optional[Any] = None
        self._model_infer_failed = False
        self._first_model_infer_debug_done = False
        self._debug_first_model_infer = os.getenv(
            "POKER44_MINER_DEBUG_FIRST_INFERENCE", "1"
        ).strip() in {"1", "true", "yes", "on"}
        self._debug_score_distribution = os.getenv(
            "POKER44_MINER_DEBUG_SCORE_DISTRIBUTION", "0"
        ).strip() in {"1", "true", "yes", "on"}
        self._score_sat_eps = float(
            os.getenv("POKER44_MINER_DEBUG_SCORE_SAT_EPS", "0.01").strip() or "0.01"
        )
        self._request_counter = 0
        if self._request_logging_enabled:
            self._request_log_path.parent.mkdir(parents=True, exist_ok=True)
            bt.logging.info(
                f"📥 Miner request logging enabled → {self._request_log_path} "
                f"(full_chunks={self._request_logging_full_chunks}, "
                f"ssl_export={self._request_log_ssl_export}, "
                f"chunk_ndjson={self._request_log_chunk_ndjson}, "
                f"slim_export={self._request_log_slim_export}, "
                f"rotate_every_s={self._request_log_rotate_seconds}, async={self._log_async})"
            )
            if self._log_async:
                self._start_async_log_worker()
        self._maybe_load_model()
        self._maybe_load_wgz_meta()
        self._maybe_load_transform_meta()
        self._maybe_load_ssl_artifact()
        self._maybe_load_dbf_bounds()
        _tm_path = self._resolve_transform_meta_path()
        _wgz_path = self._resolve_wgz_meta_path()
        _dbf_path = self._resolve_dbf_bounds_path()
        bt.logging.info(
            f"Miner scoring stack: model_path={self._model_path or ''!r} "
            f"loaded={bool(self._model)} | transform_meta_path={str(_tm_path) if _tm_path else ''!r} "
            f"loaded={bool(self._transform_meta)} | "
            f"wgz_meta_path={str(_wgz_path) if _wgz_path else ''!r} "
            f"loaded={bool(self._wgz_meta)} | "
            f"ssl_npz={self._ssl_npz_path or ''!r} loaded={bool(self._ssl_art)} | "
            f"dbf_bounds_path={str(_dbf_path) if _dbf_path else ''!r} "
            f"loaded={bool(self._dbf_quantile_bounds)} "
            f"expects_dbf={self._model_expects_dbf_features()}"
        )
        self._warn_ssl_model_mismatch()
        self._maybe_build_feature_hash_map()
        if self._miner_require_model and self._model is None:
            raise RuntimeError(
                "POKER44_MINER_REQUIRE_MODEL is enabled but no estimator loaded. "
                "Set POKER44_MINER_MODEL_PATH to a valid joblib file (path must exist and load)."
            )
        if self._miner_require_model:
            bt.logging.info("POKER44_MINER_REQUIRE_MODEL=1 — heuristic fallback disabled.")
        atexit.register(self._shutdown_log_worker)

        _repo_commit_default = _git_head_sha()
        if not _repo_commit_default.strip() and not (
            os.getenv("POKER44_MODEL_REPO_COMMIT") or ""
        ).strip():
            bt.logging.warning(
                "No git HEAD and POKER44_MODEL_REPO_COMMIT unset — subnet validators require "
                "repo_commit in model_manifest (integrity: manifest_missing_repo_commit)."
            )

        self.model_manifest = build_local_model_manifest(
            repo_root=REPO_ROOT,
            implementation_files=[Path(__file__).resolve()],
            defaults={
                "model_name": "poker44-reference-heuristic",
                "model_version": "1",
                "framework": "python-heuristic",
                "license": "MIT",
                "repo_url": "https://github.com/Poker44/Poker44-subnet",
                "repo_commit": _repo_commit_default,
                "notes": "Reference heuristic miner shipped with the Poker44 subnet.",
                "open_source": True,
                "inference_mode": "remote",
                "training_data_statement": (
                    "Reference heuristic miner. No training step. Uses only runtime chunk features."
                ),
                "training_data_sources": ["none"],
                "private_data_attestation": (
                    "This reference miner does not train on validator-only evaluation data."
                ),
            },
        )
        self.manifest_compliance = evaluate_manifest_compliance(self.model_manifest)
        self.manifest_digest = manifest_digest(self.model_manifest)
        self._log_manifest_startup(REPO_ROOT)
        
        # # Attach handlers after initialization
        # self.axon.attach(
        #     forward_fn = self.forward,
        #     blacklist_fn = self.blacklist,
        #     priority_fn = self.priority,
        # )
        # bt.logging.info("Attaching forward function to miner axon.")

        bt.logging.info(f"Axon created: {self.axon}")

    def _log_manifest_startup(self, repo_root: Path) -> None:
        bt.logging.info("Open-sourced miner manifest standard active for this miner.")
        bt.logging.info(
            f"Miner transparency status: {self.manifest_compliance['status']} "
            f"(missing_fields={self.manifest_compliance['missing_fields']})"
        )
        bt.logging.info(
            f"Manifest summary | model={self.model_manifest.get('model_name', '')} "
            f"version={self.model_manifest.get('model_version', '')} "
            f"repo={self.model_manifest.get('repo_url', '')} "
            f"commit={self.model_manifest.get('repo_commit', '')} "
            f"open_source={self.model_manifest.get('open_source')}"
        )
        bt.logging.info(
            f"Manifest digest={self.manifest_digest} "
            f"inference_mode={self.model_manifest.get('inference_mode', '')}"
        )
        bt.logging.info(
            "Miner prep docs available | "
            f"miner_doc={repo_root / 'docs' / 'miner.md'}"
        )

    async def forward(self, synapse: DetectionSynapse) -> DetectionSynapse:
        """Assign one deterministic bot-risk score per chunk."""
        chunks = synapse.chunks or []
        t0 = time.perf_counter()
        raw_scores = [float(self._score_chunk_runtime(chunk)) for chunk in chunks]
        smoothed_scores = self._smooth_scores_if_enabled(raw_scores)
        # Monotonic remap (no-op unless POKER44_SCORE_REMAP_PATH points to an
        # active config). Applied after smoothing so logged raw_scores and the
        # ScoreMonitor still see the model's native distribution.
        final_scores = self._remap_scores_if_enabled(smoothed_scores)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        vhk = self._validator_hotkey_from_synapse(synapse)
        self._record_latency(latency_ms)
        synapse.risk_scores = final_scores
        synapse.predictions = [s >= self._inference_threshold for s in final_scores]
        synapse.model_manifest = dict(self.model_manifest)
        self._maybe_log_request(
            synapse=synapse,
            chunks=chunks,
            scores=final_scores,
            scores_raw=raw_scores,
            predictions=synapse.predictions,
            latency_ms=latency_ms,
        )
        self._maybe_log_score_diagnostics(
            chunks=chunks,
            scores=final_scores,
            scores_raw=raw_scores,
            validator_hotkey=vhk,
        )
        # ScoreMonitor watches the model's native (pre-remap) distribution so
        # bimodality detection is not affected by the remap.
        self._score_monitor.add_scores(smoothed_scores)
        bt.logging.info(
            "Miner predictions (True=bot, False=human vs "
            f"threshold={self._inference_threshold:g}): {synapse.predictions}"
        )
        bt.logging.info(
            f"Scored {len(chunks)} chunks in {latency_ms:.2f}ms "
            f"({int(sum(len(c) for c in chunks))} hands)."
        )
        bt.logging.info(f"Validator Hotkey: {vhk}")
        return synapse

    def _maybe_load_model(self) -> None:
        if self._model_bundle_dir:
            bundle_dir = Path(self._model_bundle_dir).expanduser().resolve()
            model_path = bundle_dir / "lgbm_student.joblib"
            if model_path.is_file():
                self._model_path = str(model_path)
                self._maybe_load_bundle_runtime(bundle_dir)
            else:
                bt.logging.warning(
                    f"POKER44_MINER_MODEL_BUNDLE_DIR set but missing joblib: {model_path}"
                )
        if not self._model_path:
            bt.logging.info("Miner scoring mode: heuristic (POKER44_MINER_MODEL_PATH not set)")
            return
        model_path = Path(self._model_path).expanduser().resolve()
        if not model_path.exists():
            bt.logging.warning(
                f"Miner model path does not exist: {model_path}. Falling back to heuristic scoring."
            )
            return
        try:
            joblib = importlib.import_module("joblib")
            model = joblib.load(model_path)
            self._model = model
            self._model_features = self._resolve_model_feature_names(model, model_path)
            feat_info = (
                f"{len(self._model_features)} features"
                if self._model_features is not None
                else "unknown feature order"
            )
            bt.logging.info(f"Miner scoring mode: model ({model_path}, {feat_info})")
            if self._model_features is None:
                bt.logging.warning(
                    "Model feature list unknown: predict_proba will receive every key from "
                    "aggregate_chunk_from_hands (can mismatch 56-col estimators). Add "
                    "feature_cols.json next to the joblib or use an estimator that exposes "
                    "feature_names_in_ / feature_name_in_ (e.g. unwrap CalibratedClassifierCV)."
                )
        except Exception as e:
            bt.logging.warning(
                f"Failed to load miner model from {model_path}: {e}. Falling back to heuristic scoring."
            )
            self._model = None
            self._model_features = None

    def _maybe_load_bundle_runtime(self, bundle_dir: Path) -> None:
        """Load optional bundle metadata (threshold + adapter) for tbm_v5-style runtime."""
        # Inference threshold from retrain summary (fallback 0.5).
        summary_path = bundle_dir / "retrain_summary.json"
        if summary_path.is_file():
            try:
                payload = json.loads(summary_path.read_text(encoding="utf-8"))
                self._inference_threshold = float(payload.get("selected_threshold", 0.5))
            except Exception as e:
                bt.logging.warning(f"Failed to read bundle retrain_summary.json ({e}); threshold=0.5")
                self._inference_threshold = 0.5
        # Adapter runtime for adp_* feature synthesis.
        adp_path = bundle_dir / "adapter" / "dl_adapter.pt"
        if adp_path.is_file():
            try:
                torch = importlib.import_module("torch")
                try:
                    payload = torch.load(adp_path, map_location="cpu", weights_only=False)
                except TypeError:
                    # Backward compatibility for torch versions without weights_only kwarg.
                    payload = torch.load(adp_path, map_location="cpu")
                nn = importlib.import_module("torch.nn")
                self._tbm_adapter = {
                    "feature_cols": [str(x) for x in payload["feature_cols"]],
                    "mean": np.asarray(payload["mean"], dtype=np.float32),
                    "std": np.asarray(payload["std"], dtype=np.float32),
                    "hidden_dim": int(payload["hidden_dim"]),
                    "embed_dim": int(payload["embed_dim"]),
                    "dropout": float(payload["dropout"]),
                    "state_dict": payload["state_dict"],
                }
                enc = self._build_adapter_encoder(
                    nn,
                    len(self._tbm_adapter["feature_cols"]),
                    int(self._tbm_adapter["hidden_dim"]),
                    int(self._tbm_adapter["embed_dim"]),
                    float(self._tbm_adapter["dropout"]),
                )
                state_dict = self._tbm_adapter["state_dict"]
                enc_state = {
                    k[len("encoder.") :]: v
                    for k, v in state_dict.items()
                    if str(k).startswith("encoder.")
                }
                enc.load_state_dict(enc_state)
                enc.eval()
                self._tbm_adapter_encoder = enc
                bt.logging.info(
                    f"Loaded tbm adapter runtime: {adp_path} (embed_dim={self._tbm_adapter['embed_dim']})"
                )
            except Exception as e:
                bt.logging.warning(f"Failed to load tbm adapter runtime ({adp_path}): {e}")
                self._tbm_adapter = None
                self._tbm_adapter_encoder = None
        else:
            self._tbm_adapter = None
            self._tbm_adapter_encoder = None

    def _resolve_transform_meta_path(self) -> Optional[Path]:
        # Explicit env override wins.
        if self._transform_meta_path:
            return Path(self._transform_meta_path).expanduser().resolve()
        # Common colocated artifact: same folder as model.
        if self._model_path:
            model_path = Path(self._model_path).expanduser().resolve()
            return model_path.with_name("transform_meta.json")
        return None

    def _resolve_wgz_meta_path(self) -> Optional[Path]:
        # Explicit env override wins.
        if self._wgz_meta_path:
            return Path(self._wgz_meta_path).expanduser().resolve()
        # Bundle-local artifact.
        if self._model_bundle_dir:
            p = Path(self._model_bundle_dir).expanduser().resolve() / "within_group_zscore_meta.json"
            return p
        # Common colocated artifact: same folder as model.
        if self._model_path:
            model_path = Path(self._model_path).expanduser().resolve()
            return model_path.with_name("within_group_zscore_meta.json")
        return None

    def _maybe_load_wgz_meta(self) -> None:
        path = self._resolve_wgz_meta_path()
        if path is None:
            bt.logging.info("WGZ transform disabled (no path configured).")
            return
        if not path.exists():
            bt.logging.info(f"WGZ meta not found at {path}; skipping WGZ transform.")
            return
        try:
            with path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            if not isinstance(meta, dict):
                raise ValueError("within_group_zscore_meta root must be a JSON object.")
            if "group_col" not in meta or "global" not in meta:
                raise ValueError("WGZ meta missing required keys: group_col/global")
            self._wgz_meta = meta
            bt.logging.info(f"Loaded WGZ meta: {path}")
        except Exception as e:
            bt.logging.warning(f"Failed to load WGZ meta from {path}: {e}. WGZ transform disabled.")
            self._wgz_meta = None

    def _maybe_load_transform_meta(self) -> None:
        path = self._resolve_transform_meta_path()
        if path is None:
            bt.logging.info("Transform meta disabled (no path configured).")
            return
        if not path.exists():
            bt.logging.info(f"Transform meta not found at {path}; using raw aggregate features.")
            return
        try:
            with path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            if not isinstance(meta, dict):
                raise ValueError("transform_meta root must be a JSON object.")
            self._transform_meta = meta
            bt.logging.info(f"Loaded transform meta: {path}")
        except Exception as e:
            bt.logging.warning(
                f"Failed to load transform meta from {path}: {e}. Using raw aggregate features."
            )
            self._transform_meta = None

    @staticmethod
    def _wgz_group_key(v: Any) -> Optional[int]:
        try:
            x = float(v)
        except Exception:
            return None
        if not np.isfinite(x):
            return None
        x = float(np.clip(np.round(x), 2.0, 10.0))
        return int(x)

    def _apply_runtime_wgz(self, row: dict[str, Any]) -> dict[str, float]:
        meta = self._wgz_meta
        if not meta:
            return {k: self._safe_float(v, 0.0) for k, v in row.items()}
        out: dict[str, float] = {k: self._safe_float(v, np.nan) for k, v in row.items()}
        group_col = str(meta.get("group_col", "n_players_max"))
        global_stats = meta.get("global", {}) or {}
        per_group = meta.get("per_group", {}) or {}
        cols = meta.get("normalize_cols", []) or []

        gk = self._wgz_group_key(out.get(group_col, np.nan))
        gdict = per_group.get(str(gk), {}) if gk is not None and isinstance(per_group, dict) else {}
        for c in cols:
            if c not in out:
                continue
            x = out.get(c, np.nan)
            if not np.isfinite(x):
                continue
            st = gdict.get(c) if isinstance(gdict, dict) else None
            if not isinstance(st, dict):
                st = global_stats.get(c, {"mean": 0.0, "std": 1.0})
            mu = self._safe_float(st.get("mean", 0.0), 0.0)
            sig = self._safe_float(st.get("std", 1.0), 1.0)
            if abs(sig) < 1e-12:
                sig = 1.0
            out[c] = float((x - mu) / sig)
        for feat, x in list(out.items()):
            if not np.isfinite(x):
                out[feat] = 0.0
        return out

    def _maybe_load_ssl_artifact(self) -> None:
        if not self._ssl_npz_path:
            return
        path = Path(self._ssl_npz_path).expanduser().resolve()
        if not path.is_file():
            bt.logging.warning(
                f"POKER44_MINER_SSL_NPZ_PATH set but file missing: {path} "
                "(emb_* columns will be zeros if the model expects them)."
            )
            return
        try:
            from poker44.utils.ssl_embed_runtime import load_ssl_masked_ae

            self._ssl_art = load_ssl_masked_ae(path)
            n_in = len(self._ssl_art["feature_cols"])
            bt.logging.info(f"Loaded SSL masked-AE artifact: {path} (encoder_in_features={n_in})")
        except Exception as e:
            bt.logging.warning(f"Failed to load SSL artifact {path}: {e}")
            self._ssl_art = None

    def _resolve_dbf_bounds_path(self) -> Optional[Path]:
        if self._dbf_bounds_path_env:
            return Path(self._dbf_bounds_path_env).expanduser().resolve()
        if self._model_path:
            return Path(self._model_path).expanduser().resolve().with_name("dbf_quantile_bounds.json")
        return None

    def _model_expects_dbf_features(self) -> bool:
        if not self._model_features:
            return False
        return any(str(f).startswith("dbf_") for f in self._model_features)

    def _maybe_load_dbf_bounds(self) -> None:
        """Load train-fitted DBF winsor bounds when the estimator lists ``dbf_*`` inputs."""
        self._dbf_quantile_bounds = None
        if not self._model or not self._model_expects_dbf_features():
            return
        path = self._resolve_dbf_bounds_path()
        if path is None or not path.is_file():
            msg = (
                "Model expects dbf_* features but DBF bounds JSON was not found. "
                f"Set POKER44_MINER_DBF_BOUNDS_JSON or place dbf_quantile_bounds.json next to the joblib "
                f"(tried {path})."
            )
            if self._miner_require_model:
                raise RuntimeError(msg)
            bt.logging.warning(msg + " Using 0.0 for all dbf_* at inference.")
            return
        try:
            from poker44.utils.dbf_chunk_runtime import load_dbf_quantile_bounds_file

            self._dbf_quantile_bounds = load_dbf_quantile_bounds_file(REPO_ROOT, path)
            n = len(self._dbf_quantile_bounds)
            bt.logging.info(f"Loaded DBF quantile bounds: {path} ({n} dbf columns)")
            self._warn_dbf_transform_meta_overlap()
        except Exception as e:
            if self._miner_require_model:
                raise RuntimeError(f"Failed to load DBF bounds from {path}: {e}") from e
            bt.logging.warning(f"Failed to load DBF bounds ({e}); dbf_* will be 0.0.")

    def _warn_dbf_transform_meta_overlap(self) -> None:
        """If meta ever lists ``dbf_*``, clip/log/scale could double-process DBF — warn once."""
        meta = self._transform_meta
        if not meta or not isinstance(meta, dict):
            return
        clip = meta.get("clip_bounds") or {}
        if isinstance(clip, dict) and any(str(k).startswith("dbf_") for k in clip):
            bt.logging.warning(
                "transform_meta clip_bounds includes dbf_* keys. Miner applies meta only before "
                "DBF; dbf_* are not re-clipped by meta. Confirm this matches your training script."
            )
        log_feats = meta.get("log1p_selected_features") or []
        if isinstance(log_feats, list) and any(str(k).startswith("dbf_") for k in log_feats):
            bt.logging.warning(
                "transform_meta log1p_selected_features includes dbf_* — miner does not log1p "
                "dbf_* after compute_dbf_frame; verify training parity."
            )
        scale_stats = meta.get("robust_scale_stats") or {}
        if isinstance(scale_stats, dict) and any(str(k).startswith("dbf_") for k in scale_stats):
            bt.logging.warning(
                "transform_meta robust_scale_stats includes dbf_* — miner does not robust-scale "
                "dbf_* via meta; verify training parity."
            )

    def _append_dbf_features_after_transform(self, row: dict[str, float]) -> dict[str, float]:
        """
        Append ``dbf_*`` after ``transform_meta`` — matches ``*_with_dbf`` parquet pipeline
        (base columns robust-transformed first; DBF uses train-fitted quantiles only).

        ``transform_meta`` must not re-apply to ``dbf_*`` (those keys are absent from
        explorer ``transform_meta.json``); DBF already applies winsor + tanh internally.
        """
        if not self._model_expects_dbf_features():
            return row
        if self._dbf_quantile_bounds:
            try:
                from poker44.utils.dbf_chunk_runtime import compute_dbf_row_values

                dbf_vals = compute_dbf_row_values(REPO_ROOT, row, self._dbf_quantile_bounds)
                merged = dict(row)
                merged.update(dbf_vals)
                return merged
            except Exception as e:
                if self._miner_require_model:
                    raise RuntimeError(f"DBF feature compute failed: {e}") from e
                bt.logging.warning(f"DBF compute failed ({e}); using 0.0 for dbf_*.")
        out = dict(row)
        for k in self._model_features or []:
            sk = str(k)
            if sk.startswith("dbf_"):
                out[sk] = 0.0
        return out

    def _model_expects_ssl_embeddings(self) -> bool:
        if not self._model_features:
            return False
        return any(str(f).startswith("emb_") for f in self._model_features)

    def _model_expects_adapter_embeddings(self) -> bool:
        if not self._model_features:
            return False
        return any(str(f).startswith("adp_") for f in self._model_features)

    def _warn_ssl_model_mismatch(self) -> None:
        if self._model_expects_adapter_embeddings() and self._tbm_adapter is None:
            bt.logging.warning(
                "Model feature list includes adp_* but tbm adapter runtime is not loaded. "
                "Set POKER44_MINER_MODEL_BUNDLE_DIR to a valid bundle (with adapter/dl_adapter.pt)."
            )
        needs = self._model_expects_ssl_embeddings()
        has = self._ssl_art is not None
        if needs and not has:
            bt.logging.warning(
                "Model feature list includes emb_* but POKER44_MINER_SSL_NPZ_PATH is unset or failed to load. "
                "Those inputs will be passed as 0.0 — set the npz path to match training."
            )
        elif has and not needs:
            bt.logging.info(
                "SSL masked-AE artifact loaded but model feature_cols have no emb_* — encoder is unused."
            )
        elif has and needs and self._ssl_art is not None and self._model_features is not None:
            try:
                from poker44.utils.ssl_embed_runtime import ssl_embedding_from_row

                z, _ = ssl_embedding_from_row({}, self._ssl_art)
                zdim = int(z.shape[1])
                emb_idx = []
                for f in self._model_features:
                    if not str(f).startswith("emb_"):
                        continue
                    try:
                        emb_idx.append(int(str(f)[4:]))
                    except ValueError:
                        continue
                if emb_idx:
                    need_dim = max(emb_idx) + 1
                    if need_dim != zdim:
                        bt.logging.warning(
                            f"SSL embedding dim mismatch: model expects emb_* up to index {need_dim - 1} "
                            f"({need_dim} dims) but encoder outputs {zdim}. Check artifact vs joblib."
                        )
            except Exception as e:
                bt.logging.warning(f"Could not validate SSL embedding width: {e}")

    def _maybe_build_feature_hash_map(self) -> None:
        """Validate that feature obfuscation is active and log status.

        The actual mapping is applied on-the-fly at inference time by hashing
        each raw feature key with the salt.  ``self._model_features`` already
        contains hashed names (from the hashed ``feature_cols.json``), so we
        only need to confirm alignment here, not pre-build a mapping table.
        """
        if not self._feature_salt:
            self._feature_hash_map = None
            return
        if not self._model_features:
            self._feature_hash_map = None
            return
        first_feat = self._model_features[0] if self._model_features else ""
        looks_hashed = first_feat.startswith("f_") and len(first_feat) == 12
        if not looks_hashed:
            bt.logging.warning(
                "POKER44_FEATURE_SALT is set but feature_cols.json does not appear hashed "
                f"(first feature: {first_feat!r}). Obfuscation will be active — make sure "
                "bundle feature names are hashed."
            )
        self._feature_hash_map = True  # sentinel: on-the-fly hashing enabled
        bt.logging.info(
            f"Feature obfuscation active: {len(self._model_features)} features "
            f"(salt len={len(self._feature_salt)})"
        )

    @staticmethod
    def _build_adapter_encoder(nn: Any, in_dim: int, hidden_dim: int, embed_dim: int, dropout: float) -> Any:
        return nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

    def _augment_row_with_tbm_adapter(self, row: dict[str, float]) -> dict[str, float]:
        """Append adp_* features from bundle adapter for model runtime parity."""
        if not self._model_expects_adapter_embeddings():
            return row
        if self._tbm_adapter is None:
            if self._miner_require_model:
                raise RuntimeError("Model expects adp_* but tbm adapter runtime is unavailable.")
            out = dict(row)
            for k in self._model_features or []:
                if str(k).startswith("adp_"):
                    out[str(k)] = 0.0
            return out
        try:
            torch = importlib.import_module("torch")
            feat_cols = self._tbm_adapter["feature_cols"]
            mean = self._tbm_adapter["mean"]
            std = self._tbm_adapter["std"]
            x = np.asarray([float(row.get(c, 0.0)) for c in feat_cols], dtype=np.float32)
            x = (x - mean) / std
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            enc = self._tbm_adapter_encoder
            if enc is None:
                raise RuntimeError("tbm adapter encoder not initialized")
            with torch.no_grad():
                z = enc(torch.from_numpy(x.reshape(1, -1))).detach().cpu().numpy().astype(np.float32)
            out = dict(row)
            for i in range(z.shape[1]):
                out[f"adp_{i:03d}"] = float(z[0, i])
            return out
        except Exception as e:
            if self._miner_require_model:
                raise RuntimeError(f"TBM adapter embedding failed: {e}") from e
            bt.logging.warning(f"TBM adapter embedding failed ({e}); using 0.0 for adp_*.")
            out = dict(row)
            for k in self._model_features or []:
                if str(k).startswith("adp_"):
                    out[str(k)] = 0.0
            return out

    @staticmethod
    def _signed_log1p(x: float) -> float:
        # Mirrors training transform behavior for potentially negative values.
        return float(np.sign(x) * np.log1p(np.abs(x)))

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            v = float(value)
            if not np.isfinite(v):
                return default
            return v
        except Exception:
            return default

    def _apply_runtime_transform_meta(self, row: dict[str, Any]) -> dict[str, float]:
        meta = self._transform_meta
        if not meta:
            return {k: self._safe_float(v, 0.0) for k, v in row.items()}

        out: dict[str, float] = {k: self._safe_float(v, np.nan) for k, v in row.items()}

        clip_cfg = meta.get("clip", {}) or {}
        clip_bounds = meta.get("clip_bounds", {}) or {}
        if bool(clip_cfg.get("enabled", False)) and isinstance(clip_bounds, dict):
            for feat, bounds in clip_bounds.items():
                if not isinstance(bounds, dict):
                    continue
                if feat not in out:
                    continue
                x = out.get(feat, np.nan)
                if not np.isfinite(x):
                    continue
                lo = bounds.get("low", None)
                hi = bounds.get("high", None)
                if lo is not None:
                    x = max(x, self._safe_float(lo, x))
                if hi is not None:
                    x = min(x, self._safe_float(hi, x))
                out[feat] = float(x)

        log_cfg = meta.get("log1p", {}) or {}
        log_feats = meta.get("log1p_selected_features", []) or []
        if bool(log_cfg.get("enabled", False)) and isinstance(log_feats, list):
            for feat in log_feats:
                if feat not in out:
                    continue
                x = out.get(feat, np.nan)
                if np.isfinite(x):
                    out[feat] = self._signed_log1p(float(x))

        scale_cfg = meta.get("robust_scale", {}) or {}
        scale_stats = meta.get("robust_scale_stats", {}) or {}
        if bool(scale_cfg.get("enabled", False)) and isinstance(scale_stats, dict):
            for feat, st in scale_stats.items():
                if feat not in out or not isinstance(st, dict):
                    continue
                x = out.get(feat, np.nan)
                if not np.isfinite(x):
                    continue
                median = self._safe_float(st.get("median", 0.0), 0.0)
                iqr = self._safe_float(st.get("iqr", 1.0), 1.0)
                if abs(iqr) < 1e-12:
                    iqr = 1.0
                out[feat] = float((x - median) / iqr)

            scaled_clip_abs = scale_cfg.get("scaled_clip_abs", None)
            if scaled_clip_abs is not None:
                sca = abs(self._safe_float(scaled_clip_abs, 0.0))
                if sca > 0:
                    for feat in scale_stats.keys():
                        if feat not in out:
                            continue
                        x = out.get(feat, np.nan)
                        if np.isfinite(x):
                            out[feat] = float(np.clip(x, -sca, sca))

        fill_cfg = meta.get("fillna", {}) or {}
        medians = fill_cfg.get("medians", {}) or {}
        if isinstance(medians, dict):
            for feat, med in medians.items():
                x = out.get(feat, np.nan)
                if not np.isfinite(x):
                    out[feat] = self._safe_float(med, 0.0)

        for feat, x in list(out.items()):
            if not np.isfinite(x):
                out[feat] = 0.0

        return out

    @staticmethod
    def _base_estimator_for_feature_names(model: Any) -> Any:
        """Unwrap Pipeline / CalibratedClassifierCV to the underlying GBDT (same idea as cross_dataset_eval)."""
        steps = getattr(model, "steps", None)
        if isinstance(steps, list) and len(steps) > 0:
            return Miner._base_estimator_for_feature_names(steps[-1][1])
        ccl = getattr(model, "calibrated_classifiers_", None)
        if isinstance(ccl, list) and len(ccl) > 0:
            inner = ccl[0]
            est = getattr(inner, "estimator", None)
            if est is not None:
                return Miner._base_estimator_for_feature_names(est)
        return model

    @staticmethod
    def _feature_cols_from_artifact_dir(model_path: Path) -> Optional[list[str]]:
        """``feature_cols.json`` written by training scripts (calibration / lgbm)."""
        fc_path = model_path.parent / "feature_cols.json"
        if not fc_path.is_file():
            return None
        try:
            payload = json.loads(fc_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        cols = payload.get("feature_cols")
        if not isinstance(cols, list) or not cols:
            return None
        return [str(c) for c in cols if str(c)]

    @classmethod
    def _resolve_model_feature_names(cls, model: Any, model_path: Path) -> Optional[list[str]]:
        """
        Column order for ``predict_proba`` — must match training (e.g. 56 cols), not the full
        aggregate_chunk_from_hands key set (often larger as the schema grows).

        Validators send **hand JSON** in ``synapse.chunks``; features are computed locally.
        Only these names are passed to the estimator when this list is non-None.
        """
        from_json = cls._feature_cols_from_artifact_dir(model_path)
        if from_json:
            return from_json

        for m in (model, cls._base_estimator_for_feature_names(model)):
            for attr in ("feature_names_in_", "feature_name_in_"):
                names = getattr(m, attr, None)
                if names is not None and len(names) > 0:
                    return [str(x) for x in names]
            booster = getattr(m, "booster_", None)
            if booster is not None:
                try:
                    raw = booster.feature_name()
                    if raw:
                        return [str(x) for x in raw]
                except Exception:
                    pass
        return None

    def _score_chunk_runtime(self, chunk: list[dict]) -> float:
        """
        Chunk dicts from the validator are never passed to the model.

        Flow: ``chunk`` (hands JSON) → :func:`aggregate_chunk_from_hands` →
        ``_apply_runtime_transform_meta`` (base columns only in meta) → optional
        ``dbf_*`` via train-fitted bounds → optional ``emb_*`` → align to
        ``feature_cols`` → ``predict_proba``.
        """
        if self._other_only_mode:
            return self._score_chunk_other_only(chunk)

        if self._miner_require_model:
            if self._model is None:
                raise RuntimeError("POKER44_MINER_REQUIRE_MODEL but self._model is None")
            if self._model_infer_failed:
                raise RuntimeError(
                    "Model inference failed earlier; restart miner after fixing the estimator."
                )

        if self._model is not None and not self._model_infer_failed:
            try:
                raw_row = aggregate_chunk_from_hands(chunk, skip_sanitize=True)
                raw_row = _compute_invariant_features(raw_row)
                row = self._apply_runtime_wgz(raw_row)
                if self._feature_hash_map and self._feature_salt:
                    row = {
                        _hash_feature_name(k, self._feature_salt): v
                        for k, v in row.items()
                    }
                row = self._apply_runtime_transform_meta(row)
                row = self._append_dbf_features_after_transform(row)
                row = self._augment_row_with_tbm_adapter(row)
                if not row:
                    if self._miner_require_model:
                        raise ValueError(
                            "aggregate_chunk_from_hands returned no features; cannot run model."
                        )
                    return 0.5
                row_for_model: dict[str, float] = row
                if (
                    self._ssl_art is not None
                    and self._model_features is not None
                    and self._model_expects_ssl_embeddings()
                ):
                    from poker44.utils.ssl_embed_runtime import augment_row_with_ssl_embeddings

                    row_for_model = augment_row_with_ssl_embeddings(
                        row, self._ssl_art, self._model_features
                    )
                if self._model_features is None:
                    X = pd.DataFrame([row_for_model])
                else:
                    aligned = {k: float(row_for_model.get(k, 0.0)) for k in self._model_features}
                    X = pd.DataFrame([aligned], columns=self._model_features)
                # Hard guarantee for GBDT: only finite floats reach the estimator.
                X = X.astype(np.float64).values
                proba = self._model.predict_proba(X)
                if self._debug_first_model_infer and not self._first_model_infer_debug_done:
                    self._first_model_infer_debug_done = True
                    xflat = np.asarray(X, dtype=np.float64).ravel()
                    n_nan = int(np.isnan(xflat).sum())
                    n_inf = int(np.isinf(xflat).sum())
                    feat_names = self._model_features or []
                    missing = [k for k in feat_names if k not in row_for_model]
                    classes = getattr(self._model, "classes_", None)
                    cls_repr = (
                        np.asarray(classes).tolist()
                        if classes is not None
                        else None
                    )
                    bt.logging.info(
                        "First model inference debug: "
                        f"classes_={cls_repr} "
                        f"proba_row={np.asarray(proba[0]).tolist()} "
                        f"X_shape={tuple(X.shape)} n_nan={n_nan} n_inf={n_inf} "
                        f"x_min={float(np.nanmin(xflat)):.6g} x_max={float(np.nanmax(xflat)):.6g} "
                        f"x_mean={float(np.nanmean(xflat)):.6g} "
                        f"aggregate_keys={len(row)} model_features={len(feat_names)} "
                        f"missing_keys_sample={missing[:12]}"
                    )
                score = float(proba[0][1])
                return round(self._clamp01(score), 6)
            except Exception as e:
                if self._miner_require_model:
                    bt.logging.error(f"Model inference failed (POKER44_MINER_REQUIRE_MODEL=1): {e}")
                    raise
                self._model_infer_failed = True
                bt.logging.warning(
                    f"Model inference failed once ({e}); switching to heuristic scoring."
                )
        return self.score_chunk(chunk)

    def _score_chunk_other_only(self, chunk: list[dict]) -> float:
        """Pure 'other' signal rule: human chunks contain 'other' actions, bots don't."""
        if not chunk:
            return 0.5
        row = aggregate_chunk_from_hands(chunk, skip_sanitize=True)
        if not row:
            return 0.5
        other_ratio_mean = float(row.get("other_ratio_mean", 0.0))
        return 0.0 if other_ratio_mean > 0 else 1.0

    def _start_async_log_worker(self) -> None:
        if self._log_queue is not None and self._log_worker is not None:
            return
        self._log_queue = queue.Queue(maxsize=max(100, self._log_queue_max))
        self._log_worker_stop.clear()
        self._log_worker = threading.Thread(
            target=self._log_worker_loop,
            name="miner-log-writer",
            daemon=True,
        )
        self._log_worker.start()
        bt.logging.info(
            f"🧵 Async log worker started (queue_max={self._log_queue.maxsize}, "
            f"batch={self._log_batch_size}, flush_ms={self._log_flush_ms})"
        )

    def _shutdown_log_worker(self) -> None:
        worker = self._log_worker
        q = self._log_queue
        if worker is None or q is None:
            return
        try:
            self._log_worker_stop.set()
            q.put_nowait({"_kind": "stop"})
        except Exception:
            pass
        worker.join(timeout=2.0)

    def _effective_log_path(self) -> Path:
        path = self._request_log_path
        interval_s = getattr(self, "_request_log_rotate_seconds", None)
        if interval_s and interval_s > 0:
            elapsed = max(0.0, time.monotonic() - getattr(self, "_log_rotate_anchor", 0.0))
            idx = int(elapsed // interval_s)
            path = path.with_name(f"{path.stem}_{idx}{path.suffix}")
        return path

    def _write_log_rows(self, rows: list[dict]) -> None:
        if not rows:
            return
        path = self._effective_log_path()
        with path.open("a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False))
                f.write("\n")

    def _log_worker_loop(self) -> None:
        assert self._log_queue is not None
        q = self._log_queue
        buffer: list[dict] = []
        flush_every_s = max(0.05, self._log_flush_ms / 1000.0)
        next_flush = time.monotonic() + flush_every_s
        while not self._log_worker_stop.is_set():
            timeout = max(0.01, next_flush - time.monotonic())
            try:
                item = q.get(timeout=timeout)
                if item.get("_kind") == "stop":
                    break
                if item.get("_kind") == "row":
                    buffer.append(item["row"])
                if (
                    len(buffer) >= max(1, self._log_batch_size)
                    or self._pending_flush_signal
                    or time.monotonic() >= next_flush
                ):
                    self._write_log_rows(buffer)
                    buffer.clear()
                    self._pending_flush_signal = False
                    next_flush = time.monotonic() + flush_every_s
            except queue.Empty:
                if buffer:
                    self._write_log_rows(buffer)
                    buffer.clear()
                next_flush = time.monotonic() + flush_every_s
            except Exception as e:
                bt.logging.warning(f"Async log worker error: {e}")
        if buffer:
            try:
                self._write_log_rows(buffer)
            except Exception as e:
                bt.logging.warning(f"Final async log flush failed: {e}")

    def _record_latency(self, latency_ms: float) -> None:
        self._latency_hist.append(float(latency_ms))
        if not self._adaptive_logging:
            self._log_state = "normal"
            return
        if len(self._latency_hist) < 20:
            return
        arr = sorted(self._latency_hist)
        idx = int(0.95 * (len(arr) - 1))
        p95 = arr[idx]
        high_ms = self._timeout_budget_ms * self._high_load_ratio
        extreme_ms = self._timeout_budget_ms * self._extreme_load_ratio
        if p95 >= extreme_ms:
            self._log_state = "extreme"
        elif p95 >= high_ms:
            self._log_state = "high"
        else:
            self._log_state = "normal"

    def _should_log_full_chunks_now(self) -> bool:
        if not self._request_logging_full_chunks:
            return False
        if not self._adaptive_logging:
            return True
        return self._log_state == "normal"

    def _build_log_row(
        self,
        *,
        synapse: DetectionSynapse,
        chunks: list[list[dict]],
        scores: list[float],
        predictions: list[bool],
        latency_ms: float,
        scores_raw: Optional[list[float]] = None,
    ) -> Optional[dict]:
        if self._request_log_ssl_export:
            if not chunks:
                return None
            if not self._should_log_full_chunks_now():
                return None
            row_ssl: dict[str, Any] = {
                "chunk_hashes": [self._stable_chunk_hash(c) for c in chunks],
                "chunks": chunks,
            }
            if scores_raw is not None and len(scores_raw) == len(chunks):
                row_ssl["risk_scores_raw"] = [float(s) for s in scores_raw]
            return row_ssl
        if self._request_log_slim_export:
            if not chunks:
                return None
            if not self._should_log_full_chunks_now():
                return None
            return {"chunks": chunks}

        if self._adaptive_logging and self._log_state == "extreme":
            # Minimal counters-only row in extreme mode.
            return {
                "ts_utc": datetime.now(tz=timezone.utc).isoformat(),
                "request_idx": self._request_counter,
                "log_state": self._log_state,
                "n_chunks": len(chunks),
                "total_hands": int(sum(len(c) for c in chunks)),
                "latency_ms": round(float(latency_ms), 3),
                "dropped_log_rows": int(self._dropped_log_rows),
            }

        row = {
            "ts_utc": datetime.now(tz=timezone.utc).isoformat(),
            "request_idx": self._request_counter,
            "log_state": self._log_state,
            "n_chunks": len(chunks),
            "total_hands": int(sum(len(c) for c in chunks)),
            "latency_ms": round(float(latency_ms), 3),
            "chunk_sizes": [len(c) for c in chunks],
            "chunk_hashes": [self._stable_chunk_hash(c) for c in chunks],
            # risk_scores = what the validator receives (after uncertain-band smoothing)
            "risk_scores": [float(s) for s in scores],
            "predictions": [bool(p) for p in predictions],
        }
        if scores_raw is not None and len(scores_raw) == len(scores):
            row["risk_scores_raw"] = [float(s) for s in scores_raw]
        vhk = self._validator_hotkey_from_synapse(synapse)
        if vhk:
            row["validator_hotkey"] = vhk
        if self._should_log_full_chunks_now():
            row["chunks"] = chunks
        return row

    def _enqueue_log_row(self, row: dict) -> None:
        q = self._log_queue
        if q is None:
            return
        try:
            q.put_nowait({"_kind": "row", "row": row})
        except queue.Full:
            self._dropped_log_rows += 1
            return
        qsize = q.qsize()
        qmax = max(1, q.maxsize)
        if qsize / qmax >= self._log_near_full_ratio:
            # Ask worker to flush soon; never block scoring path.
            self._pending_flush_signal = True

    @staticmethod
    def _smooth_uncertain_score(s: float, a: float, b: float, gamma: float) -> float:
        if s <= a or s >= b:
            return s
        z = (s - a) / (b - a)
        z_soft = z ** gamma
        return a + (b - a) * z_soft

    def _smooth_scores_if_enabled(self, scores: list[float]) -> list[float]:
        a = self._uncertain_a
        b = self._uncertain_b
        gamma = self._uncertain_gamma
        if not (0.0 <= a < b <= 1.0) or gamma <= 0.0:
            return scores
        return [self._clamp01(self._smooth_uncertain_score(float(s), a, b, gamma)) for s in scores]

    def _maybe_load_score_remap(self) -> None:
        """Load a piecewise-linear monotonic remap config (anchors)."""
        self._score_remap_anchors = None
        if not self._score_remap_path:
            return
        path = Path(self._score_remap_path).expanduser().resolve()
        if not path.is_file():
            bt.logging.warning(f"POKER44_SCORE_REMAP_PATH set but file missing: {path}")
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            bt.logging.warning(f"Failed to parse score remap config {path}: {e}")
            return
        if not bool(payload.get("active", False)):
            bt.logging.info(
                f"Score remap config {path} loaded but inactive (reason="
                f"{payload.get('reason', '')!r}); scores will not be remapped."
            )
            return
        anchors_raw = payload.get("anchors")
        if not isinstance(anchors_raw, list) or len(anchors_raw) < 2:
            bt.logging.warning(f"Score remap config {path} has invalid anchors; ignoring.")
            return
        anchors: list[tuple[float, float]] = []
        last_x = -1.0
        last_y = -1.0
        for pair in anchors_raw:
            try:
                x = float(pair[0])
                y = float(pair[1])
            except (TypeError, ValueError, IndexError):
                bt.logging.warning(f"Score remap config {path} has malformed anchor {pair!r}.")
                return
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                bt.logging.warning(f"Score remap anchor out of [0,1]: {pair!r}")
                return
            if x < last_x or y < last_y:
                bt.logging.warning(
                    f"Score remap anchors not monotonic at {pair!r} "
                    f"(last_x={last_x}, last_y={last_y}); ignoring."
                )
                return
            anchors.append((x, y))
            last_x, last_y = x, y
        self._score_remap_anchors = anchors
        bt.logging.info(
            f"Score remap loaded ({len(anchors)} anchors) from {path}: {anchors}"
        )

    def _apply_score_remap(self, s: float) -> float:
        anchors = self._score_remap_anchors
        if not anchors:
            return s
        s = 0.0 if s < 0.0 else (1.0 if s > 1.0 else s)
        if s <= anchors[0][0]:
            return anchors[0][1]
        if s >= anchors[-1][0]:
            return anchors[-1][1]
        for i in range(len(anchors) - 1):
            x0, y0 = anchors[i]
            x1, y1 = anchors[i + 1]
            if x0 <= s <= x1:
                if x1 <= x0:
                    return y1
                t = (s - x0) / (x1 - x0)
                return y0 + t * (y1 - y0)
        return s

    def _remap_scores_if_enabled(self, scores: list[float]) -> list[float]:
        if not self._score_remap_anchors:
            return scores
        return [self._clamp01(self._apply_score_remap(float(s))) for s in scores]

    @staticmethod
    def _stable_chunk_hash(chunk: list[dict]) -> str:
        payload = json.dumps(
            chunk,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _validator_hotkey_from_synapse(synapse: DetectionSynapse) -> Optional[str]:
        d = getattr(synapse, "dendrite", None)
        if d is None:
            return None
        hk = getattr(d, "hotkey", None)
        return str(hk) if hk else None

    def _maybe_log_request(
        self,
        *,
        synapse: DetectionSynapse,
        chunks: list[list[dict]],
        scores: list[float],
        predictions: list[bool],
        latency_ms: float,
        scores_raw: Optional[list[float]] = None,
    ) -> None:
        """Best-effort request logging; never raise into scoring path."""
        if not self._request_logging_enabled:
            return
        try:
            self._request_counter += 1
            if self._request_log_chunk_ndjson:
                if chunks and self._should_log_full_chunks_now():
                    use_raw = (
                        scores_raw is not None
                        and len(scores_raw) == len(chunks)
                    )
                    nd_rows: list[dict[str, Any]] = []
                    for i, ch in enumerate(chunks):
                        rs = (
                            float(scores_raw[i])
                            if use_raw
                            else float(scores[i])
                        )
                        nd_rows.append(
                            {
                                "chunk_hash": self._stable_chunk_hash(ch),
                                "chunk": ch,
                                "risk_score": rs,
                            }
                        )
                    if nd_rows:
                        if self._log_async:
                            for r in nd_rows:
                                self._enqueue_log_row(r)
                        else:
                            self._write_log_rows(nd_rows)
                        return
                # NDJSON enabled but chunks skipped (adaptive/high load) or empty:
                # fall through so ssl/slim/full rows still record counters or scores.
            row = self._build_log_row(
                synapse=synapse,
                chunks=chunks,
                scores=scores,
                predictions=predictions,
                latency_ms=latency_ms,
                scores_raw=scores_raw,
            )
            if row is None:
                return
            if self._log_async:
                self._enqueue_log_row(row)
            else:
                self._write_log_rows([row])
        except Exception as e:
            bt.logging.warning(f"Failed to log miner request: {e}")

    def _maybe_log_score_diagnostics(
        self,
        *,
        chunks: list[list[dict]],
        scores: list[float],
        scores_raw: Optional[list[float]],
        validator_hotkey: Optional[str],
    ) -> None:
        """Optional per-request diagnostics for score spread and repeated patterns."""
        if not self._debug_score_distribution or not scores:
            return
        try:
            arr = np.asarray(scores, dtype=np.float64)
            eps = max(0.0, float(self._score_sat_eps))
            near_floor = int(np.sum(arr <= eps))
            near_ceil = int(np.sum(arr >= (1.0 - eps)))
            bool_vec = "".join("1" if float(s) >= self._inference_threshold else "0" for s in arr.tolist())
            pred_sig = hashlib.sha256(bool_vec.encode("utf-8")).hexdigest()[:12]
            chunk_hashes_sample = [self._stable_chunk_hash(c)[:12] for c in chunks[:3]]
            raw_mean = float(np.mean(scores_raw)) if scores_raw else float("nan")
            bt.logging.info(
                "Score diagnostics: "
                f"n={len(scores)} min={float(np.min(arr)):.6f} max={float(np.max(arr)):.6f} "
                f"mean={float(np.mean(arr)):.6f} std={float(np.std(arr)):.6f} "
                f"near_floor={near_floor} near_ceil={near_ceil} eps={eps:.4f} "
                f"pred_sig={pred_sig} chunk_hashes_head={chunk_hashes_sample} "
                f"raw_mean={raw_mean:.6f} validator={validator_hotkey}"
            )
        except Exception as e:
            bt.logging.warning(f"Score diagnostics failed: {e}")

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, value))

    @classmethod
    def _score_chunk_row(cls, row: dict) -> float:
        """
        Map one chunk-level feature row (same keys as LGBM training) to [0, 1].

        Uses aggregated means only; std/max are in ``row`` for richer models.
        """
        street_depth = min(1.0, float(row.get("n_streets_mean", 0.0)) / 3.0)
        call_mean = float(row.get("call_ratio_mean", 0.0))
        check_mean = float(row.get("check_ratio_mean", 0.0))
        fold_mean = float(row.get("fold_ratio_mean", 0.0))
        raise_mean = float(row.get("raise_ratio_mean", 0.0))
        npm = float(row.get("n_players_mean", 0.0))
        player_count_signal = (6 - min(npm, 6)) / 4.0 if npm else 0.0

        score = 0.0
        score += 0.32 * street_depth
        score += 0.18 * cls._clamp01(call_mean / 0.35)
        score += 0.12 * cls._clamp01(check_mean / 0.30)
        score += 0.08 * cls._clamp01(player_count_signal)
        score -= 0.18 * cls._clamp01(fold_mean / 0.55)
        score -= 0.10 * cls._clamp01(raise_mean / 0.20)
        return cls._clamp01(score)

    @classmethod
    def score_chunk(cls, chunk: list[dict]) -> float:
        if not chunk:
            return 0.5

        row = aggregate_chunk_from_hands(chunk, skip_sanitize=True)
        if not row:
            return 0.5

        return round(cls._clamp01(cls._score_chunk_row(row)), 6)

    async def blacklist(self, synapse: DetectionSynapse) -> Tuple[bool, str]:
        """Determine whether to blacklist incoming requests."""
        return self.common_blacklist(synapse)

    async def priority(self, synapse: DetectionSynapse) -> float:
        """Assign priority based on caller's stake."""
        return self.caller_priority(synapse)


if __name__ == "__main__":
    with Miner() as miner:
        bt.logging.info("Random miner running...")
        while True:
            bt.logging.info(f"Miner UID: {miner.uid} | Incentive: {miner.metagraph.I[miner.uid]}")
            time.sleep(5 * 60)
