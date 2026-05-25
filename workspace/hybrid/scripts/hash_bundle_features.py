"""Hash feature names in a model bundle for IP protection.

Replaces all feature names in feature_cols.json and transform_meta.json
with salted hashes. The model joblib is NOT modified — RF uses positions,
not names.

Usage:
    python hash_bundle_features.py --bundle <path> --salt <secret>
    python hash_bundle_features.py --bundle <path> --salt-env POKER44_FEATURE_SALT
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def hash_feature_name(name: str, salt: str) -> str:
    raw = hashlib.sha256((salt + name).encode("utf-8")).hexdigest()[:10]
    return f"f_{raw}"


def build_mapping(feature_cols: list[str], salt: str) -> dict[str, str]:
    mapping = {}
    for name in feature_cols:
        hashed = hash_feature_name(name, salt)
        if hashed in mapping.values():
            raise ValueError(f"Hash collision: {name} -> {hashed}")
        mapping[name] = hashed
    return mapping


def hash_dict_keys(d: dict, mapping: dict[str, str]) -> dict:
    """Replace keys in a dict using the mapping. Non-matching keys kept as-is."""
    return {mapping.get(k, k): v for k, v in d.items()}


def hash_transform_meta(meta: dict, mapping: dict[str, str]) -> dict:
    out = dict(meta)

    if "clip_bounds" in out and isinstance(out["clip_bounds"], dict):
        out["clip_bounds"] = hash_dict_keys(out["clip_bounds"], mapping)

    if "log1p_selected_features" in out and isinstance(out["log1p_selected_features"], list):
        out["log1p_selected_features"] = [mapping.get(f, f) for f in out["log1p_selected_features"]]

    if "robust_scale_stats" in out and isinstance(out["robust_scale_stats"], dict):
        out["robust_scale_stats"] = hash_dict_keys(out["robust_scale_stats"], mapping)

    if "fillna" in out and isinstance(out["fillna"], dict):
        fillna = dict(out["fillna"])
        if "medians" in fillna and isinstance(fillna["medians"], dict):
            fillna["medians"] = hash_dict_keys(fillna["medians"], mapping)
        out["fillna"] = fillna

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True, help="Path to model bundle directory")
    ap.add_argument("--salt", default="", help="Salt string for hashing")
    ap.add_argument("--salt-env", default="", help="Env var name containing salt")
    ap.add_argument("--dry-run", action="store_true", help="Print mapping without writing")
    args = ap.parse_args()

    import os
    salt = args.salt or os.environ.get(args.salt_env, "")
    if not salt:
        raise ValueError("Provide --salt or --salt-env with a non-empty value")

    bundle = Path(args.bundle).resolve()
    fc_path = bundle / "feature_cols.json"
    tm_path = bundle / "transform_meta.json"

    if not fc_path.is_file():
        raise FileNotFoundError(fc_path)

    fc_data = json.loads(fc_path.read_text())
    feature_cols = fc_data["feature_cols"]
    mapping = build_mapping(feature_cols, salt)

    if args.dry_run:
        print("Mapping ({} features):".format(len(mapping)))
        for real, hashed in sorted(mapping.items()):
            print("  {} -> {}".format(real, hashed))
        return

    hashed_cols = [mapping[f] for f in feature_cols]
    fc_data["feature_cols"] = hashed_cols
    fc_path.write_text(json.dumps(fc_data, indent=2))
    print("Hashed feature_cols.json ({} features)".format(len(hashed_cols)))

    if tm_path.is_file():
        meta = json.loads(tm_path.read_text())
        meta = hash_transform_meta(meta, mapping)
        tm_path.write_text(json.dumps(meta, indent=2))
        print("Hashed transform_meta.json")

    mapping_path = bundle / "feature_mapping.json"
    mapping_path.write_text(json.dumps(mapping, indent=2))
    print("Saved mapping to {} (DO NOT COMMIT THIS FILE)".format(mapping_path))
    print("\nDone. Add feature_mapping.json to .gitignore!")


if __name__ == "__main__":
    main()
