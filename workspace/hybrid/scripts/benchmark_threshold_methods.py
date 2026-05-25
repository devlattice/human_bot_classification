"""Compare threshold policies on May-8 (labeled) and real_distribution (unlabeled).

Reports bot recall / human FPR on May-8 and score-band stats (e.g. % in [0.5, 1.0]) on live JSONL.

Usage:
  python workspace/hybrid/scripts/benchmark_threshold_methods.py \\
    --bundle workspace/hybrid/model_bundle
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "workspace" / "hybrid" / "scripts"))
sys.path.insert(0, str(REPO / "workspace" / "hybrid"))

import train_production_model as tpm  # noqa: E402
from chunk_pipeline import aggregate_chunk_from_miner_payload  # noqa: E402
from poker44.miner.dynamic_threshold import (  # noqa: E402
    DynamicThresholdPolicy,
    ThresholdConfig,
    ThresholdDecision,
)

REAL_DIST = REPO / "workspace" / "dataset" / "real_distribution"
DEFAULT_OUT = REPO / "workspace" / "hybrid" / "bot_system" / "data" / "threshold_method_benchmark.json"
MAY8_PATH = tpm.MAY8_GOLD_TEST_PATH
ZENODO_PATH = tpm.TEST_DIR / "zenodo_test_features.parquet"
FIXED_T = 0.5


@dataclass
class MethodSpec:
    name: str
    description: str
    # Returns threshold for this batch given scores so far
    factory: Callable[[], "ThresholdMethod"]


class ThresholdMethod:
    """Per-stream stateful threshold chooser (reset per dataset)."""

    def reset(self) -> None:
        pass

    def threshold_for_batch(self, batch_scores: list[float]) -> tuple[float, str]:
        raise NotImplementedError

    def observe_batch(self, batch_scores: list[float]) -> None:
        pass


class FixedThreshold(ThresholdMethod):
    def __init__(self, t: float) -> None:
        self.t = t

    def threshold_for_batch(self, batch_scores: list[float]) -> tuple[float, str]:
        return self.t, "fixed"


class PolicyWrapper(ThresholdMethod):
    def __init__(self, cfg: ThresholdConfig) -> None:
        self.policy = DynamicThresholdPolicy(cfg)

    def reset(self) -> None:
        self.policy = DynamicThresholdPolicy(self.policy.config)

    def threshold_for_batch(self, batch_scores: list[float]) -> tuple[float, str]:
        dec = self.policy.decide(batch_scores)
        self.policy.observe(batch_scores)
        return dec.threshold, dec.mode


class Gmm2Batch(ThresholdMethod):
    def __init__(self, clamp: tuple[float, float], min_scores: int = 20) -> None:
        self.clamp = clamp
        self.min_scores = min_scores

    def threshold_for_batch(self, batch_scores: list[float]) -> tuple[float, str]:
        if len(batch_scores) < self.min_scores:
            return FIXED_T, "gmm_fallback"
        arr = np.asarray(batch_scores, dtype=np.float64).reshape(-1, 1)
        try:
            gmm = GaussianMixture(n_components=2, random_state=42, max_iter=100)
            gmm.fit(arr)
            means = sorted(float(m) for m in gmm.means_.ravel())
            t = (means[0] + means[1]) / 2.0
            t = max(self.clamp[0], min(self.clamp[1], t))
            return t, "gmm2_batch"
        except Exception:
            return FIXED_T, "gmm_fail"


class PassiveMedian(ThresholdMethod):
    """If batch median low, use static_low; else fixed 0.5."""

    def __init__(self, static_low: float, median_cut: float = 0.35) -> None:
        self.static_low = static_low
        self.median_cut = median_cut

    def threshold_for_batch(self, batch_scores: list[float]) -> tuple[float, str]:
        if not batch_scores:
            return FIXED_T, "passive_empty"
        med = float(np.median(batch_scores))
        if med < self.median_cut:
            return self.static_low, "passive_low"
        return FIXED_T, "passive_high"


class HybridGapPassive(ThresholdMethod):
    """Gap policy first; if gap fails and median < cut → static_low."""

    def __init__(self, cfg: ThresholdConfig, median_cut: float = 0.35) -> None:
        self.cfg = cfg
        self.median_cut = median_cut
        self.static_low = cfg.static_selected or cfg.fixed_fallback
        self._policy = DynamicThresholdPolicy(cfg)

    def reset(self) -> None:
        self._policy = DynamicThresholdPolicy(self.cfg)

    def threshold_for_batch(self, batch_scores: list[float]) -> tuple[float, str]:
        dec = self._policy.decide(batch_scores)
        if dec.mode in ("gap_batch", "gap_rolling"):
            self._policy.observe(batch_scores)
            return dec.threshold, dec.mode
        med = float(np.median(batch_scores)) if batch_scores else 1.0
        if med < self.median_cut:
            self._policy.observe(batch_scores)
            return self._policy._clamp(self.static_low), "hybrid_passive"
        self._policy.observe(batch_scores)
        return dec.threshold, dec.mode


class HdbscanRolling(ThresholdMethod):
    def __init__(self, clamp: tuple[float, float], window: int = 500, min_scores: int = 30) -> None:
        self.clamp = clamp
        self.window = window
        self.min_scores = min_scores
        self._roll: deque[float] = deque(maxlen=window)

    def reset(self) -> None:
        self._roll.clear()

    def observe_batch(self, batch_scores: list[float]) -> None:
        for s in batch_scores:
            self._roll.append(float(s))

    def threshold_for_batch(self, batch_scores: list[float]) -> tuple[float, str]:
        self.observe_batch(batch_scores)
        if len(self._roll) < self.min_scores:
            return FIXED_T, "hdbscan_warmup"
        try:
            import hdbscan  # type: ignore
        except ImportError:
            return FIXED_T, "hdbscan_missing"
        arr = np.asarray(list(self._roll), dtype=np.float64).reshape(-1, 1)
        labels = hdbscan.HDBSCAN(min_cluster_size=max(10, len(arr) // 20)).fit_predict(arr)
        uniq = [lab for lab in set(labels) if lab >= 0]
        if len(uniq) < 2:
            return FIXED_T, "hdbscan_one_cluster"
        centers = []
        for lab in uniq:
            sub = arr[labels == lab]
            centers.append((float(np.median(sub)), lab))
        centers.sort(key=lambda x: x[0])
        t = (centers[0][0] + centers[-1][0]) / 2.0
        t = max(self.clamp[0], min(self.clamp[1], t))
        return t, "hdbscan_rolling"


def load_bundle(bundle: Path) -> tuple[RandomForestClassifier, list[str], dict, float | None]:
    for name in ("model.joblib", "lgbm_student.joblib"):
        mp = bundle / name
        if mp.is_file():
            rf = joblib.load(mp)
            break
    else:
        raise FileNotFoundError(f"no model in {bundle}")
    cols = json.loads((bundle / "feature_cols.json").read_text(encoding="utf-8"))["feature_cols"]
    tm = json.loads((bundle / "transform_meta.json").read_text(encoding="utf-8"))
    static_t: float | None = None
    for name in ("production_threshold.json", "retrain_summary.json"):
        p = bundle / name
        if p.is_file():
            try:
                static_t = float(json.loads(p.read_text(encoding="utf-8")).get("selected_threshold"))
                break
            except Exception:
                pass
    return rf, cols, tm, static_t


def score_df(df: pd.DataFrame, cols: list[str], rf: RandomForestClassifier, tm: dict) -> np.ndarray:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"missing {len(miss)} cols")
    Xt = tpm.apply_transform(df[cols].values, cols, tm)
    return rf.predict_proba(Xt)[:, 1].astype(np.float64)


def simulate_method(
    scores: np.ndarray,
    labels: np.ndarray | None,
    method: ThresholdMethod,
    batch_size: int,
) -> dict[str, Any]:
    method.reset()
    n = len(scores)
    pred = np.zeros(n, dtype=bool)
    thresholds: list[float] = []
    modes: list[str] = []

    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        batch = scores[start:end].tolist()
        t, mode = method.threshold_for_batch(batch)
        if not isinstance(method, PolicyWrapper) and not isinstance(method, HybridGapPassive):
            method.observe_batch(batch)
        thresholds.extend([t] * (end - start))
        modes.extend([mode] * (end - start))
        pred[start:end] = scores[start:end] >= t

    out: dict[str, Any] = {
        "n_chunks": int(n),
        "n_requests": int((n + batch_size - 1) // batch_size),
        "score_mean": round(float(scores.mean()), 4),
        "score_median": round(float(np.median(scores)), 4),
        "threshold_median": round(float(np.median(thresholds)), 4) if thresholds else None,
        "threshold_mean": round(float(np.mean(thresholds)), 4) if thresholds else None,
        "mode_counts": {str(k): int(v) for k, v in pd.Series(modes).value_counts().items()},
        "pct_scores_in_0.5_1.0": round(float(((scores >= 0.5) & (scores <= 1.0)).mean()) * 100, 2),
        "pct_pred_bot": round(float(pred.mean()) * 100, 2),
    }

    if labels is not None:
        y = labels.astype(int)
        bot, human = y == 1, y == 0
        if bot.any():
            out["bot_recall_pct"] = round(float(pred[bot].mean()) * 100, 2)
        if human.any():
            out["human_fpr_pct"] = round(float(pred[human].mean()) * 100, 3)

    return out


def build_methods(static_t: float | None) -> list[MethodSpec]:
    st = static_t if static_t is not None else 0.52
    clamp = (0.10, 0.45)

    def gap_cfg(fallback: float) -> ThresholdConfig:
        return ThresholdConfig(
            fixed_fallback=FIXED_T,
            static_selected=fallback,
            clamp_min=clamp[0],
            clamp_max=clamp[1],
            min_scores=20,
            rolling_window=500,
        )

    return [
        MethodSpec("fixed_0.5", "Global 0.5", lambda: FixedThreshold(0.5)),
        MethodSpec("static_bundle", f"Fixed bundle selected ({st:g})", lambda: FixedThreshold(st)),
        MethodSpec("static_0.18", "Fixed 0.18 (May-8-oriented sweep ref)", lambda: FixedThreshold(0.18)),
        MethodSpec(
            "gap_dynamic",
            "Bimodal gap + rolling; static fallback from bundle",
            lambda: PolicyWrapper(gap_cfg(st)),
        ),
        MethodSpec(
            "gap_static_0.18",
            "Bimodal gap; static fallback 0.18",
            lambda: PolicyWrapper(gap_cfg(0.18)),
        ),
        MethodSpec("gmm2_batch", "2-Gaussian midpoint per batch", lambda: Gmm2Batch(clamp)),
        MethodSpec("passive_median_0.18", "median<0.35 → 0.18 else 0.5", lambda: PassiveMedian(0.18)),
        MethodSpec(
            "hybrid_gap_passive",
            "Gap → else passive low → else policy fallback",
            lambda: HybridGapPassive(gap_cfg(st)),
        ),
        MethodSpec("hdbscan_rolling", "HDBSCAN on rolling scores (optional dep)", lambda: HdbscanRolling(clamp)),
    ]


def pick_winner(results: dict[str, Any], *, max_human_fpr: float = 1.0) -> dict[str, Any]:
    """Rank by May-8 bot recall with FPR caps on May-8 human + zenodo."""
    rows: list[dict[str, Any]] = []
    for mname, block in results.get("methods", {}).items():
        may8 = block.get("may8_all", {})
        may8h = block.get("may8_human", {})
        zen = block.get("zenodo_test", {})
        rec = may8.get("bot_recall_pct")
        fpr_m = may8h.get("human_fpr_pct")
        fpr_z = zen.get("human_fpr_pct")
        if rec is None:
            continue
        ok = True
        if fpr_m is not None and fpr_m > max_human_fpr:
            ok = False
        if fpr_z is not None and fpr_z > 2.0:
            ok = False
        rows.append({
            "method": mname,
            "may8_bot_recall_pct": rec,
            "may8_human_fpr_pct": fpr_m,
            "zenodo_fpr_pct": fpr_z,
            "passes_gates": ok,
            "may8_hard_recall": block.get("may8_hard", {}).get("bot_recall_pct"),
        })
    passing = [r for r in rows if r["passes_gates"]]
    pool = passing if passing else rows
    pool.sort(key=lambda r: (-(r["may8_bot_recall_pct"] or 0), r.get("may8_human_fpr_pct") or 999))
    return {"ranking": rows, "recommended": pool[0]["method"] if pool else None}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, default=REPO / "workspace" / "hybrid" / "model_bundle")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--batch-size", type=int, default=20)
    p.add_argument("--real-dist-dir", type=Path, default=REAL_DIST)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rf, cols, tm, static_t = load_bundle(args.bundle)
    specs = build_methods(static_t)

    may8 = pd.read_parquet(MAY8_PATH)
    may8_b = may8[may8["label"] == 1]
    may8_h = may8[may8["label"] == 0]

    # hard/easy split via reference model on gold train
    gold_train = pd.read_parquet(tpm.GOLD_PATH)
    zen = pd.read_parquet(ZENODO_PATH) if ZENODO_PATH.is_file() else None
    ref_h_parts = [gold_train[gold_train["label"] == 0][cols]]
    ref_b_parts = [gold_train[gold_train["label"] == 1][cols]]
    if zen is not None:
        ref_h_parts.append(
            zen.sample(n=min(len(zen), tpm.HUMAN_SAMPLE_CAP), random_state=args.seed)[cols]
        )
    Xrh = pd.concat(ref_h_parts, ignore_index=True)
    Xrb = pd.concat(ref_b_parts, ignore_index=True)
    Xr = pd.concat([Xrh, Xrb], ignore_index=True).values
    yr = np.concatenate([np.zeros(len(Xrh)), np.ones(len(Xrb))])
    Xrt, ref_tm = tpm.fit_transform_pipeline(Xr, cols)
    ref_rf = RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=15,
        random_state=args.seed, n_jobs=-1, class_weight="balanced",
    )
    ref_rf.fit(Xrt, yr)
    pb = ref_rf.predict_proba(tpm.apply_transform(may8_b[cols].values, cols, ref_tm))[:, 1]
    may8_hard = may8_b.loc[pb < 0.5]
    may8_easy = may8_b.loc[pb >= 0.5]

    results: dict[str, Any] = {
        "bundle": str(args.bundle),
        "batch_size": args.batch_size,
        "static_from_bundle": static_t,
        "note_real_dist": (
            "pct_scores_in_0.5_1.0 = fraction of model scores in [0.5,1] (unlabeled). "
            "pct_pred_bot = fraction flagged bot at chosen threshold. "
            "Use May-8 recall/FPR to pick method; real_dist is monitoring only."
        ),
        "methods": {},
    }

    datasets: list[tuple[str, pd.DataFrame, np.ndarray | None]] = [
        ("may8_all", may8, may8["label"].values),
        ("may8_human", may8_h, may8_h["label"].values),
        ("may8_bot_all", may8_b, may8_b["label"].values),
        ("may8_hard", may8_hard, may8_hard["label"].values),
        ("may8_easy", may8_easy, may8_easy["label"].values),
    ]
    if ZENODO_PATH.is_file():
        zdf = pd.read_parquet(ZENODO_PATH)
        datasets.append(("zenodo_test", zdf, zdf["label"].values))

    print("=" * 72)
    print(f"THRESHOLD BENCHMARK  bundle={args.bundle.name}  batch={args.batch_size}")
    print("=" * 72)

    for spec in specs:
        print(f"\n--- {spec.name}: {spec.description} ---")
        method = spec.factory()
        block: dict[str, Any] = {}

        for dname, df, labels in datasets:
            scores = score_df(df, cols, rf, tm)
            block[dname] = simulate_method(scores, labels, method, args.batch_size)
            if dname == "may8_bot_all":
                print(
                    f"  may8_bot recall={block[dname].get('bot_recall_pct')}%  "
                    f"hard={block.get('may8_hard', {}).get('bot_recall_pct')}%  "
                    f"human_fpr={block.get('may8_human', {}).get('human_fpr_pct')}%"
                )

        # real_distribution
        scores_list: list[float] = []
        for fp in sorted(args.real_dist_dir.glob("*.jsonl")):
            with fp.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        chunk = obj.get("chunk")
                        if not isinstance(chunk, list):
                            continue
                        raw = aggregate_chunk_from_miner_payload(chunk)
                        if not raw or any(c not in raw for c in cols):
                            continue
                        Xv = tpm.apply_transform(
                            np.asarray([raw[c] for c in cols], dtype=np.float64)[None, :],
                            cols, tm,
                        )
                        scores_list.append(float(rf.predict_proba(Xv)[0, 1]))
                    except Exception:
                        continue

        if scores_list:
            rd_scores = np.asarray(scores_list, dtype=np.float64)
            block["real_distribution"] = simulate_method(rd_scores, None, spec.factory(), args.batch_size)
            rd = block["real_distribution"]
            print(
                f"  real_dist n={rd['n_chunks']}  "
                f"scores in [0.5,1]={rd['pct_scores_in_0.5_1.0']}%  "
                f"pred_bot={rd['pct_pred_bot']}%  thr_med={rd['threshold_median']}"
            )
        else:
            block["real_distribution"] = {"error": "no scores"}

        results["methods"][spec.name] = block

    winner = pick_winner(results)
    results["winner"] = winner
    print("\n" + "=" * 72)
    print(f"RECOMMENDED (May-8 recall, FPR gates): {winner.get('recommended')}")
    for r in winner.get("ranking", [])[:5]:
        print(
            f"  {r['method']:22s}  may8_rec={r['may8_bot_recall_pct']}%  "
            f"may8_fpr={r['may8_human_fpr_pct']}%  zen_fpr={r['zenodo_fpr_pct']}%  "
            f"hard={r.get('may8_hard_recall')}%  ok={r['passes_gates']}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[done] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
