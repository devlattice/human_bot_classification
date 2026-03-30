"""Reference Poker44 miner with simple chunk-level behavioral heuristics."""

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import bittensor as bt

from poker44.base.miner import BaseMinerNeuron
from poker44.validator.chunk_features import aggregate_chunk_from_hands
from poker44.validator.synapse import DetectionSynapse

REPO_ROOT = Path(__file__).resolve().parents[1]


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
        log_path_env = os.getenv(
            "POKER44_MINER_LOG_PATH",
            str(REPO_ROOT / "workspace" / "real_distribution" / "miner_requests.jsonl"),
        )
        self._request_log_path = Path(log_path_env).expanduser().resolve()
        self._request_log_daily = os.getenv(
            "POKER44_MINER_LOG_DAILY_ROTATE", "1"
        ).strip() in {"1", "true", "yes", "on"}
        self._request_counter = 0
        if self._request_logging_enabled:
            self._request_log_path.parent.mkdir(parents=True, exist_ok=True)
            bt.logging.info(
                f"📥 Miner request logging enabled → {self._request_log_path} "
                f"(full_chunks={self._request_logging_full_chunks}, "
                f"daily_rotate={self._request_log_daily})"
            )

        # # Attach handlers after initialization
        # self.axon.attach(
        #     forward_fn = self.forward,
        #     blacklist_fn = self.blacklist,
        #     priority_fn = self.priority,
        # )
        # bt.logging.info("Attaching forward function to miner axon.")

        bt.logging.info(f"Axon created: {self.axon}")

    async def forward(self, synapse: DetectionSynapse) -> DetectionSynapse:
        """Assign one deterministic bot-risk score per chunk."""
        chunks = synapse.chunks or []
        t0 = time.perf_counter()
        scores = [self.score_chunk(chunk) for chunk in chunks]
        latency_ms = (time.perf_counter() - t0) * 1000.0
        synapse.risk_scores = scores
        synapse.predictions = [s >= 0.5 for s in scores]
        self._maybe_log_request(
            synapse=synapse,
            chunks=chunks,
            scores=scores,
            predictions=synapse.predictions,
            latency_ms=latency_ms,
        )
        bt.logging.info(f"Miner Predctions: {synapse.predictions}")
        bt.logging.info(
            f"Scored {len(chunks)} chunks in {latency_ms:.2f}ms "
            f"({int(sum(len(c) for c in chunks))} hands)."
        )
        return synapse

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
    ) -> None:
        """Best-effort request logging; never raise into scoring path."""
        if not self._request_logging_enabled:
            return
        try:
            self._request_counter += 1
            n_chunks = len(chunks)
            total_hands = int(sum(len(c) for c in chunks))
            row = {
                "ts_utc": datetime.now(tz=timezone.utc).isoformat(),
                "request_idx": self._request_counter,
                "n_chunks": n_chunks,
                "total_hands": total_hands,
                "latency_ms": round(float(latency_ms), 3),
                "chunk_sizes": [len(c) for c in chunks],
                "chunk_hashes": [self._stable_chunk_hash(c) for c in chunks],
                "risk_scores": [float(s) for s in scores],
                "predictions": [bool(p) for p in predictions],
            }
            vhk = self._validator_hotkey_from_synapse(synapse)
            if vhk:
                row["validator_hotkey"] = vhk
            if self._request_logging_full_chunks:
                row["chunks"] = chunks
            path = self._request_log_path
            if self._request_log_daily:
                day = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
                path = path.with_name(f"{path.stem}-{day}{path.suffix}")
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False))
                f.write("\n")
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
