#!/usr/bin/env python3
"""
Probe domain shift by predicting data source from features.

If a model can easily tell source domains apart, you likely have domain
fingerprint features that can cause collapse in cross-domain evaluation.

Supports:
  1) Legacy binary mode: --source-a / --source-b
  2) Multi-domain mode: repeated --source NAME=PATH (repeat NAME to add files)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import defaultdict
from typing import DefaultDict, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize


def _read_table(path: Path) -> pd.DataFrame:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    suf = path.suffix.lower()
    if suf == ".parquet":
        return pd.read_parquet(path)
    if suf == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"{path}: expected .parquet or .csv")


def _load_source(paths: List[Path], name: str) -> pd.DataFrame:
    if not paths:
        raise ValueError(f"{name}: no files provided")
    frames = [_read_table(p) for p in paths]
    df = pd.concat(frames, axis=0, ignore_index=True)
    if df.empty:
        raise ValueError(f"{name}: loaded dataframe is empty")
    return df


def _infer_features(all_sources: Dict[str, pd.DataFrame], label_col: str) -> List[str]:
    names = list(all_sources.keys())
    if len(names) < 2:
        raise ValueError("Need at least 2 source domains")
    shared = set(all_sources[names[0]].columns) - {label_col}
    for n in names[1:]:
        shared &= set(all_sources[n].columns) - {label_col}
    shared = sorted(shared)
    if not shared:
        raise ValueError("No shared feature columns across all sources")
    return shared


def _cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    va = np.var(a, ddof=1)
    vb = np.var(b, ddof=1)
    pooled = ((len(a) - 1) * va + (len(b) - 1) * vb) / (len(a) + len(b) - 2)
    if pooled <= 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / np.sqrt(pooled))


def _single_feature_shift_scores_multidomain(
    source_frames: Dict[str, pd.DataFrame], features: List[str]
) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    source_names = list(source_frames.keys())
    y_parts: List[np.ndarray] = []
    x_parts_by_feature: Dict[str, List[np.ndarray]] = {f: [] for f in features}
    for i, name in enumerate(source_names):
        df = source_frames[name]
        y_parts.append(np.full(len(df), i, dtype=int))
        for f in features:
            x_parts_by_feature[f].append(pd.to_numeric(df[f], errors="coerce").to_numpy(dtype=float))
    y = np.concatenate(y_parts)

    for f in features:
        parts = x_parts_by_feature[f]
        x = np.concatenate(parts)
        finite_mask = np.isfinite(x)
        if finite_mask.sum() < 4:
            auc_macro_abs = float("nan")
        else:
            yy = y[finite_mask]
            xx = x[finite_mask]
            if len(np.unique(xx)) <= 1:
                auc_macro_abs = 0.5
            else:
                # One-vs-rest AUC per class, then macro average of abs-oriented AUC.
                auc_per_class: List[float] = []
                for cls_idx in range(len(source_names)):
                    y_bin = (yy == cls_idx).astype(int)
                    if y_bin.min() == y_bin.max():
                        continue
                    auc_raw = roc_auc_score(y_bin, xx)
                    auc_per_class.append(float(max(auc_raw, 1.0 - auc_raw)))
                auc_macro_abs = float(np.mean(auc_per_class)) if auc_per_class else float("nan")

        row: Dict[str, float] = {"feature": f, "domain_auc_macro_abs": auc_macro_abs}
        means: List[float] = []
        for name, arr in zip(source_names, parts):
            finite = arr[np.isfinite(arr)]
            m = float(np.mean(finite)) if len(finite) else float("nan")
            s = float(np.std(finite)) if len(finite) else float("nan")
            row[f"mean_{name}"] = m
            row[f"std_{name}"] = s
            means.append(m)
        finite_means = [v for v in means if np.isfinite(v)]
        row["mean_range_across_sources"] = (
            float(max(finite_means) - min(finite_means)) if len(finite_means) >= 2 else float("nan")
        )

        # Max pairwise Cohen's d to highlight strongest pairwise source split.
        max_abs_d = 0.0
        seen_pair = False
        for i in range(len(parts)):
            ai = parts[i][np.isfinite(parts[i])]
            for j in range(i + 1, len(parts)):
                bj = parts[j][np.isfinite(parts[j])]
                d = _cohen_d(ai, bj)
                if np.isfinite(d):
                    seen_pair = True
                    max_abs_d = max(max_abs_d, abs(float(d)))
        row["max_abs_cohen_d_pairwise"] = float(max_abs_d) if seen_pair else float("nan")
        rows.append(row)

    out = pd.DataFrame(rows)
    out = out.sort_values(["domain_auc_macro_abs", "feature"], ascending=[False, True]).reset_index(drop=True)
    return out


def _train_domain_model(
    X: pd.DataFrame, y: np.ndarray, seed: int, test_size: float
) -> Tuple[RandomForestClassifier, float, np.ndarray, np.ndarray, np.ndarray]:
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
    model = RandomForestClassifier(
        n_estimators=700,
        max_depth=10,
        min_samples_leaf=10,
        min_samples_split=20,
        max_features="sqrt",
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    p_val = model.predict_proba(X_val)
    classes = model.classes_.astype(int)
    if len(classes) == 2:
        auc = float(roc_auc_score(y_val, p_val[:, 1]))
    else:
        y_bin = label_binarize(y_val, classes=classes)
        auc = float(roc_auc_score(y_bin, p_val, average="macro", multi_class="ovr"))
    pred_val = model.predict(X_val).astype(int)
    cm = confusion_matrix(y_val, pred_val, labels=classes)
    return model, auc, classes, cm, pred_val


def _parse_source_specs(
    source_specs: Sequence[str], source_a: Sequence[str], source_b: Sequence[str], source_a_name: str, source_b_name: str
) -> Dict[str, List[Path]]:
    out: DefaultDict[str, List[Path]] = defaultdict(list)
    if source_specs:
        for raw in source_specs:
            if "=" not in raw:
                raise ValueError(f"--source expects NAME=PATH, got: {raw!r}")
            name, p = raw.split("=", 1)
            name = name.strip()
            p = p.strip()
            if not name or not p:
                raise ValueError(f"--source expects NAME=PATH, got: {raw!r}")
            out[name].append(Path(p))
    else:
        # Backward-compatible binary mode.
        if not source_a or not source_b:
            raise ValueError("Provide either --source NAME=PATH (repeatable) or both --source-a/--source-b")
        out[source_a_name].extend(Path(p) for p in source_a)
        out[source_b_name].extend(Path(p) for p in source_b)
    if len(out) < 2:
        raise ValueError("Need at least 2 distinct source names")
    return dict(out)


def _render_feature_plots(
    out_dir: Path,
    ranked_features: List[str],
    source_names: List[str],
    source_X: Dict[str, pd.DataFrame],
    *,
    top_k: int,
    boxplot_max_points_per_source: int,
    seed: int,
) -> List[str]:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Plotting requested but matplotlib is not installed. "
            "Install it with: python -m pip install matplotlib"
        ) from e

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    emitted: List[str] = []

    top_feats = ranked_features[: max(1, min(top_k, len(ranked_features)))]
    rng = np.random.default_rng(seed)

    for feat in top_feats:
        # Mean + std bar chart by source
        means: List[float] = []
        stds: List[float] = []
        box_data: List[np.ndarray] = []
        labels: List[str] = []
        for src in source_names:
            vals = pd.to_numeric(source_X[src][feat], errors="coerce").to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            labels.append(src)
            if len(vals) == 0:
                means.append(float("nan"))
                stds.append(float("nan"))
                box_data.append(np.array([np.nan], dtype=float))
                continue
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals)))
            if len(vals) > boxplot_max_points_per_source > 0:
                idx = rng.choice(len(vals), size=boxplot_max_points_per_source, replace=False)
                vals = vals[idx]
            box_data.append(vals)

        # Bar: mean +/- std
        fig1, ax1 = plt.subplots(figsize=(8, 4.5))
        x = np.arange(len(labels))
        ax1.bar(x, means, yerr=stds, capsize=3)
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels)
        ax1.set_title(f"{feat} - mean/std by source")
        ax1.set_ylabel("value")
        fig1.tight_layout()
        p1 = plots_dir / f"{feat}_bar_mean_std.jpg"
        fig1.savefig(p1, dpi=170)
        plt.close(fig1)
        emitted.append(str(p1))

        # Boxplot: distribution by source
        fig2, ax2 = plt.subplots(figsize=(8, 4.8))
        ax2.boxplot(box_data, labels=labels, showfliers=False)
        ax2.set_title(f"{feat} - boxplot by source")
        ax2.set_ylabel("value")
        fig2.tight_layout()
        p2 = plots_dir / f"{feat}_boxplot.jpg"
        fig2.savefig(p2, dpi=170)
        plt.close(fig2)
        emitted.append(str(p2))

    return emitted


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Find domain-shift fingerprint features by source-classification."
    )
    ap.add_argument(
        "--source",
        action="append",
        default=[],
        help="Multi-domain input: NAME=PATH (.parquet/.csv). Repeat for all files/domains.",
    )
    ap.add_argument("--source-a", action="append", default=[], help="Path (.parquet/.csv), repeatable")
    ap.add_argument("--source-b", action="append", default=[], help="Path (.parquet/.csv), repeatable")
    ap.add_argument("--source-a-name", default="source_a")
    ap.add_argument("--source-b-name", default="source_b")
    ap.add_argument("--label-column", default="label", help="Ignored as feature if present")
    ap.add_argument("--sample-per-source", type=int, default=50000, help="0 = use all rows")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--out-dir",
        default="workspace/model/artifacts/domain_shift_probe",
        help="Output directory",
    )
    ap.add_argument("--top-k", type=int, default=30)
    ap.add_argument(
        "--plot-top-k",
        type=int,
        default=0,
        help="If >0, generate per-feature bar+box plots for top-k ranked features",
    )
    ap.add_argument(
        "--boxplot-max-points-per-source",
        type=int,
        default=2000,
        help="Downsample each source for boxplots (0 = no downsampling)",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    source_paths = _parse_source_specs(
        source_specs=args.source,
        source_a=args.source_a,
        source_b=args.source_b,
        source_a_name=args.source_a_name,
        source_b_name=args.source_b_name,
    )
    raw_sources: Dict[str, pd.DataFrame] = {}
    for name, paths in source_paths.items():
        raw_sources[name] = _load_source(paths, name)

    if args.sample_per_source and args.sample_per_source > 0:
        sampled_sources: Dict[str, pd.DataFrame] = {}
        for i, (name, df) in enumerate(raw_sources.items()):
            n = min(len(df), args.sample_per_source)
            sampled_sources[name] = df.sample(n=n, random_state=args.seed + i).reset_index(drop=True)
        raw_sources = sampled_sources

    feats = _infer_features(raw_sources, args.label_column)
    source_names = list(raw_sources.keys())
    source_to_id = {name: i for i, name in enumerate(source_names)}

    source_X: Dict[str, pd.DataFrame] = {}
    X_parts: List[pd.DataFrame] = []
    y_parts: List[np.ndarray] = []
    for name in source_names:
        x_df = raw_sources[name][feats].apply(pd.to_numeric, errors="coerce")
        source_X[name] = x_df
        X_parts.append(x_df)
        y_parts.append(np.full(len(x_df), source_to_id[name], dtype=int))

    X = pd.concat(X_parts, axis=0, ignore_index=True)
    y = np.concatenate(y_parts)

    # Median fill preserves ranking better than zero fill for many stats-like features.
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)

    model, domain_auc, classes, cm, _pred_val = _train_domain_model(X, y, seed=args.seed, test_size=args.test_size)
    imp = pd.DataFrame(
        {
            "feature": feats,
            "rf_importance": model.feature_importances_.astype(float),
        }
    ).sort_values(["rf_importance", "feature"], ascending=[False, True])

    uni = _single_feature_shift_scores_multidomain(source_X, feats)
    merged = uni.merge(imp, on="feature", how="left")
    merged["rank_auc"] = merged["domain_auc_macro_abs"].rank(ascending=False, method="min")
    merged["rank_rf"] = merged["rf_importance"].rank(ascending=False, method="min")
    merged["combined_rank_score"] = merged["rank_auc"] + merged["rank_rf"]
    merged = merged.sort_values(
        ["combined_rank_score", "domain_auc_macro_abs", "rf_importance"],
        ascending=[True, False, False],
    ).reset_index(drop=True)

    all_csv = out_dir / "feature_shift_report.csv"
    top_csv = out_dir / f"feature_shift_top_{args.top_k}.csv"
    summary_json = out_dir / "summary.json"
    markdown = out_dir / "report.md"

    merged.to_csv(all_csv, index=False)
    merged.head(args.top_k).to_csv(top_csv, index=False)

    class_labels = [source_names[int(c)] for c in classes]
    summary = {
        "sources": {name: int(len(source_X[name])) for name in source_names},
        "n_features": int(len(feats)),
        "domain_classifier_auc": domain_auc,
        "auc_mode": "binary_auc" if len(source_names) == 2 else "multiclass_macro_ovr_auc",
        "class_order": class_labels,
        "confusion_matrix": cm.tolist(),
        "interpretation": (
            "High AUC (close to 1.0) means strong source/domain shift. "
            "AUC near 0.5 means weak shift."
        ),
        "top_k": int(args.top_k),
        "outputs": {
            "feature_shift_report_csv": str(all_csv),
            "feature_shift_top_csv": str(top_csv),
            "report_md": str(markdown),
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    top = merged.head(args.top_k)
    lines = [
        "# Domain Shift Probe",
        "",
        "- sources:",
    ]
    for name in source_names:
        lines.append(f"  - `{name}`: {len(source_X[name])} rows")
    lines.extend(
        [
        f"- features: {len(feats)}",
        f"- domain classifier AUC: **{domain_auc:.6f}** ({summary['auc_mode']})",
        "",
        "Interpretation:",
        "- AUC >= 0.90: severe shift (likely domain fingerprints).",
        "- 0.75-0.90: moderate shift.",
        "- <= 0.60: weak shift.",
        "",
        "## Validation confusion matrix",
        "",
        f"- class order: {class_labels}",
        "",
        "| true \\ pred | " + " | ".join(class_labels) + " |",
        "|" + "---|" * (len(class_labels) + 1),
    ]
    )
    for i, row in enumerate(cm.tolist()):
        lines.append(f"| {class_labels[i]} | " + " | ".join(str(int(v)) for v in row) + " |")
    lines.extend(
        [
        "",
        f"## Top {args.top_k} domain-fingerprint features",
        "",
        "| feature | domain_auc_macro_abs | rf_importance | max_abs_cohen_d_pairwise |",
        "|---|---:|---:|---:|",
    ]
    )
    for _, r in top.iterrows():
        lines.append(
            f"| {r['feature']} | {float(r['domain_auc_macro_abs']):.6f} | "
            f"{float(r['rf_importance']):.6f} | {float(r['max_abs_cohen_d_pairwise']):.6f} |"
        )
    markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")

    plot_files: List[str] = []
    if args.plot_top_k > 0:
        plot_files = _render_feature_plots(
            out_dir,
            ranked_features=merged["feature"].tolist(),
            source_names=source_names,
            source_X=source_X,
            top_k=args.plot_top_k,
            boxplot_max_points_per_source=args.boxplot_max_points_per_source,
            seed=args.seed,
        )
        summary["plot_top_k"] = int(args.plot_top_k)
        summary["plot_files"] = plot_files
        summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("[domain_shift_probe] sources:")
    for name in source_names:
        print(f"  - {name}: rows={len(source_X[name])}")
    print(f"[domain_shift_probe] features={len(feats)}")
    print(f"[domain_shift_probe] domain_classifier_auc={domain_auc:.6f} mode={summary['auc_mode']}")
    print(f"[domain_shift_probe] wrote: {all_csv}")
    print(f"[domain_shift_probe] wrote: {top_csv}")
    print(f"[domain_shift_probe] wrote: {summary_json}")
    print(f"[domain_shift_probe] wrote: {markdown}")
    if plot_files:
        print(f"[domain_shift_probe] wrote plots: {len(plot_files)} files in {out_dir / 'plots'}")


if __name__ == "__main__":
    main()
