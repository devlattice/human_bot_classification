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


REPO_ROOT = Path(__file__).resolve().parents[1]


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
    chunk → one heuristic score in [0, 1]. The validator still receives **one risk
    score per chunk** (not per hand).

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
        self._transform_meta_path = os.getenv("POKER44_MINER_TRANSFORM_META_PATH", "").strip()
        # If true: refuse to start without a loaded model, and never fall back to heuristics
        # on a chunk (inference errors propagate; fix model or restart).
        self._miner_require_model = os.getenv(
            "POKER44_MINER_REQUIRE_MODEL", "0"
        ).strip() in {"1", "true", "yes", "on"}
        self._model: Optional[Any] = None
        self._model_features: Optional[list[str]] = None
        self._transform_meta: Optional[dict[str, Any]] = None
        self._model_infer_failed = False
        self._first_model_infer_debug_done = False
        self._debug_first_model_infer = os.getenv(
            "POKER44_MINER_DEBUG_FIRST_INFERENCE", "1"
        ).strip() in {"1", "true", "yes", "on"}
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
        self._maybe_load_transform_meta()
        _tm_path = self._resolve_transform_meta_path()
        bt.logging.info(
            "Miner scoring stack: model_path=%r loaded=%s | transform_meta_path=%r loaded=%s",
            self._model_path or "",
            bool(self._model),
            str(_tm_path) if _tm_path else "",
            bool(self._transform_meta),
        )
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
                    "This reference miner does not train on validator-private human data."
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
            "Miner prep tooling available | "
            f"benchmark_doc={repo_root / 'docs' / 'public-benchmark.md'} "
            f"miner_doc={repo_root / 'docs' / 'miner.md'} "
            f"anti_leakage_doc={repo_root / 'docs' / 'anti-leakage.md'}"
        )
        bt.logging.info(
            "Public benchmark command: "
            "python scripts/publish/publish_public_benchmark.py --skip-wandb"
        )
        bt.logging.info(
            "Purpose: train, validate and refine miner models against the public benchmark "
            "while Poker44 moves toward more dynamic evaluation."
        )

    async def forward(self, synapse: DetectionSynapse) -> DetectionSynapse:
        """Assign one deterministic bot-risk score per chunk."""
        chunks = synapse.chunks or []
        t0 = time.perf_counter()
        raw_scores = [float(self._score_chunk_runtime(chunk)) for chunk in chunks]
        smoothed_scores = self._smooth_scores_if_enabled(raw_scores)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        vhk = self._validator_hotkey_from_synapse(synapse)
        self._record_latency(latency_ms)
        synapse.risk_scores = smoothed_scores
        synapse.predictions = [s >= 0.5 for s in smoothed_scores]
        synapse.model_manifest = dict(self.model_manifest)
        self._maybe_log_request(
            synapse=synapse,
            chunks=chunks,
            scores=smoothed_scores,
            scores_raw=raw_scores,
            predictions=synapse.predictions,
            latency_ms=latency_ms,
        )
        bt.logging.info(f"Miner predictions: {synapse.predictions}")
        bt.logging.info(
            f"Scored {len(chunks)} chunks in {latency_ms:.2f}ms "
            f"({int(sum(len(c) for c in chunks))} hands)."
        )
        bt.logging.info(f"Validator Hotkey: {vhk}")
        return synapse

    def _maybe_load_model(self) -> None:
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
            self._model_features = self._model_feature_names(model)
            feat_info = (
                f"{len(self._model_features)} features"
                if self._model_features is not None
                else "unknown feature order"
            )
            bt.logging.info(f"Miner scoring mode: model ({model_path}, {feat_info})")
        except Exception as e:
            bt.logging.warning(
                f"Failed to load miner model from {model_path}: {e}. Falling back to heuristic scoring."
            )
            self._model = None
            self._model_features = None

    def _resolve_transform_meta_path(self) -> Optional[Path]:
        # Explicit env override wins.
        if self._transform_meta_path:
            return Path(self._transform_meta_path).expanduser().resolve()
        # Common colocated artifact: same folder as model.
        if self._model_path:
            model_path = Path(self._model_path).expanduser().resolve()
            return model_path.with_name("transform_meta.json")
        return None

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
    def _model_feature_names(model: Any) -> Optional[list[str]]:
        names = getattr(model, "feature_name_in_", None)
        if names is not None and len(names) > 0:
            return [str(x) for x in names]
        booster = getattr(model, "booster_", None)
        if booster is not None:
            try:
                raw = booster.feature_name()
                if raw:
                    return [str(x) for x in raw]
            except Exception:
                return None
        return None

    def _score_chunk_runtime(self, chunk: list[dict]) -> float:
        """
        Chunk dicts from the validator are never passed to the model.

        Flow: ``chunk`` (hands JSON) → :func:`aggregate_chunk_from_hands` → one numeric row
        → ``predict_proba`` on float features only.
        """
        if self._miner_require_model:
            if self._model is None:
                raise RuntimeError("POKER44_MINER_REQUIRE_MODEL but self._model is None")
            if self._model_infer_failed:
                raise RuntimeError(
                    "Model inference failed earlier; restart miner after fixing the estimator."
                )

        if self._model is not None and not self._model_infer_failed:
            try:
                raw_row = aggregate_chunk_from_hands(chunk)
                row = self._apply_runtime_transform_meta(raw_row)
                if not row:
                    if self._miner_require_model:
                        raise ValueError(
                            "aggregate_chunk_from_hands returned no features; cannot run model."
                        )
                    return 0.5
                if self._model_features is None:
                    X = pd.DataFrame([row])
                else:
                    aligned = {k: float(row.get(k, 0.0)) for k in self._model_features}
                    X = pd.DataFrame([aligned], columns=self._model_features)
                # Hard guarantee for GBDT: only finite floats reach the estimator.
                X = X.astype(np.float64)
                proba = self._model.predict_proba(X)
                if self._debug_first_model_infer and not self._first_model_infer_debug_done:
                    self._first_model_infer_debug_done = True
                    xflat = X.to_numpy(dtype=np.float64).ravel()
                    n_nan = int(np.isnan(xflat).sum())
                    n_inf = int(np.isinf(xflat).sum())
                    feat_names = self._model_features or []
                    missing = [k for k in feat_names if k not in row]
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

        row = aggregate_chunk_from_hands(chunk)
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
