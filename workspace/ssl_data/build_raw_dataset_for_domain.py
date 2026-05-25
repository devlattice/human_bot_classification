#!/usr/bin/env python3
"""
Stream JSONL miner logs → chunk-level Parquet (raw aggregates, no robust transforms).

- Reads all ``*.jsonl`` under ``--input-source-dir`` (one JSON object per line).
- Expected fields per line: ``chunk`` (list of hands), ``chunk_hash`` (dedup key),
  ``risk_score`` (ignored, not written).
- Deduplicates by ``chunk_hash`` by default (``--no-dedupe`` to keep duplicates).
- Output schema matches ``--sample`` Parquet (column names/order). Aggregator may return
  a superset of columns; only ``--sample`` feature columns are written (extras ignored).
- ``label`` is always null (validator labels are unknown here).
- Writes **one** Parquet file under ``--outdir`` (no train/val split).

CPU-friendly: line-by-line read + batched Parquet row groups (``--batch-size``).

python3 workspace/ssl_data/build_raw_dataset_for_domain.py \
  --input-source-dir workspace/ssl_data/database/source \
  --sample workspace/preprocess/statistical_test/explorer/feature_3/data/train/public/rb_B/train.parquet \
  --outdir workspace/ssl_data/raw_data/feature_3/data/validator/raw \
  --output-name validator_new.parquet

  python3 workspace/ssl_data/build_raw_dataset_for_domain.py \
  --input-sqlite-db workspace/ssl_data/database/db/chunk_hands_level.db \
  --sample workspace/preprocess/statistical_test/explorer/feature_3/data/public/train.parquet \
  --outdir workspace/ssl_data/raw_data/feature_3 \
  --output-name validator.parquet
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[2]


def _schema_with_nullable_label(sample_path: Path) -> pa.Schema:
    base = pq.read_schema(sample_path)
    fields: List[pa.Field] = []
    for f in base:
        if f.name == "label":
            fields.append(pa.field("label", f.type, nullable=True))
        else:
            fields.append(f)
    return pa.schema(fields)


def _feature_column_names(schema: pa.Schema) -> tuple[str, ...]:
    return tuple(n for n in schema.names if n != "label")


def _iter_jsonl_lines(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            yield lineno, line


def _iter_sqlite_json_rows(
    db_path: Path,
    table: str,
    line_column: str,
):
    con = sqlite3.connect(str(db_path))
    try:
        cur = con.cursor()
        query = f"SELECT {line_column} FROM {table} ORDER BY id"
        for idx, (line,) in enumerate(cur.execute(query), start=1):
            if not line:
                continue
            line = str(line).strip()
            if not line:
                continue
            yield idx, line
    finally:
        con.close()


def main() -> int:
    default_json = REPO_ROOT / "workspace" / "ssl_data" / "json"
    default_sample = (
        REPO_ROOT
        / "workspace"
        / "dataset"
        / "unpreprocessed"
        / "train"
        / "train.parquet"
    )
    default_out = REPO_ROOT / "workspace" / "ssl_data" / "raw_data"

    ap = argparse.ArgumentParser(
        description="JSONL miner logs → raw chunk-level Parquet (validator-domain features, unlabeled)."
    )
    ap.add_argument(
        "--input-source-dir",
        type=Path,
        default=default_json,
        help=f"Directory of *.jsonl (default: {default_json})",
    )
    ap.add_argument(
        "--input-sqlite-db",
        type=Path,
        default=None,
        help=(
            "Optional SQLite DB source. If set, rows are read from table/column "
            "instead of --input-source-dir JSONL files."
        ),
    )
    ap.add_argument(
        "--sqlite-table",
        default="jsonl_rows",
        help="SQLite table name containing JSONL rows (default: jsonl_rows)",
    )
    ap.add_argument(
        "--sqlite-line-column",
        default="line",
        help="SQLite column containing raw JSON text (default: line)",
    )
    ap.add_argument(
        "--sample",
        type=Path,
        default=default_sample,
        help=(
            "Reference Parquet for output columns (order + dtypes). "
            "May be full raw train (~79 features) or a keep-only table (e.g. 44); "
            "aggregator output must include every sample feature name."
        ),
    )
    ap.add_argument(
        "--outdir",
        type=Path,
        default=default_out,
        help=f"Output directory (default: {default_out})",
    )
    ap.add_argument(
        "--output-name",
        default="validator_request.parquet",
        help="Output parquet filename under outdir (default: validator_request.parquet)",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=2048,
        help="Rows per Parquet row group (default: 2048)",
    )
    ap.add_argument(
        "--log-every",
        type=int,
        default=5000,
        help="Print progress every N accepted rows (0 = quiet except file boundaries)",
    )
    ap.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Do not deduplicate by chunk_hash (default: dedupe on chunk_hash)",
    )
    args = ap.parse_args()

    input_dir = args.input_source_dir.expanduser().resolve()
    sqlite_db = (
        args.input_sqlite_db.expanduser().resolve() if args.input_sqlite_db else None
    )
    sample_path = args.sample.expanduser().resolve()
    outdir = args.outdir.expanduser().resolve()

    if sqlite_db is None and not input_dir.is_dir():
        print(f"[build_raw_domain] error: not a directory: {input_dir}", file=sys.stderr)
        return 1
    if sqlite_db is not None and not sqlite_db.is_file():
        print(f"[build_raw_domain] error: sqlite db missing: {sqlite_db}", file=sys.stderr)
        return 1
    if not sample_path.is_file():
        print(f"[build_raw_domain] error: sample parquet missing: {sample_path}", file=sys.stderr)
        return 1

    schema = _schema_with_nullable_label(sample_path)
    feature_names = _feature_column_names(schema)
    feature_set = set(feature_names)

    sys.path.insert(0, str(REPO_ROOT))
    from poker44.validator.chunk_features import aggregate_chunk_from_hands

    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / args.output_name

    jsonl_files = [] if sqlite_db is not None else sorted(input_dir.glob("*.jsonl"))
    if sqlite_db is None and not jsonl_files:
        print(f"[build_raw_domain] error: no *.jsonl under {input_dir}", file=sys.stderr)
        return 1

    seen_hash: Set[str] = set()
    batch: List[Dict[str, Any]] = []
    writer: pq.ParquetWriter | None = None

    total_lines = 0
    total_skipped_empty = 0
    n_dup = 0
    n_bad_json = 0
    n_bad_chunk = 0
    n_missing_hash = 0
    n_feat_mismatch = 0
    accepted = 0

    def flush() -> None:
        nonlocal writer, batch
        if not batch:
            return
        table = pa.Table.from_pylist(batch, schema=schema)
        if writer is None:
            writer = pq.ParquetWriter(out_path, schema, compression="snappy")
        writer.write_table(table)
        batch.clear()

    print(f"[build_raw_domain] sample_schema={sample_path}")
    print(f"[build_raw_domain] features={len(feature_names)} + label(null)")
    print(f"[build_raw_domain] dedupe_chunk_hash={not args.no_dedupe}")
    if sqlite_db is not None:
        print(
            "[build_raw_domain] sqlite_source="
            f"{sqlite_db} table={args.sqlite_table} line_column={args.sqlite_line_column}"
        )
    else:
        print(f"[build_raw_domain] input_dir={input_dir} files={len(jsonl_files)}")
    print(f"[build_raw_domain] out={out_path} batch_size={args.batch_size}")

    if sqlite_db is not None:
        print("[build_raw_domain] reading rows from sqlite ...")
        source_iter = _iter_sqlite_json_rows(
            sqlite_db,
            args.sqlite_table,
            args.sqlite_line_column,
        )
        source_items = [("sqlite", source_iter)]
    else:
        source_items = [(jpath.name, _iter_jsonl_lines(jpath)) for jpath in jsonl_files]

    for source_name, source_iter in source_items:
        print(f"[build_raw_domain] file {source_name} ...")
        for lineno, line in source_iter:
            total_lines += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                n_bad_json += 1
                continue
            if not isinstance(rec, dict):
                n_bad_chunk += 1
                continue

            ch = rec.get("chunk")
            if not isinstance(ch, list) or not ch:
                total_skipped_empty += 1
                continue

            h_raw = rec.get("chunk_hash")
            if not args.no_dedupe:
                if h_raw is None or (isinstance(h_raw, str) and not h_raw.strip()):
                    n_missing_hash += 1
                    continue
                h = str(h_raw)
                if h in seen_hash:
                    n_dup += 1
                    continue
                seen_hash.add(h)

            feat = aggregate_chunk_from_hands(ch)
            if not feat:
                n_bad_chunk += 1
                continue
            feat_keys = set(feat.keys())
            if not feature_set.issubset(feat_keys):
                n_feat_mismatch += 1
                if n_feat_mismatch <= 3:
                    missing = feature_set - feat_keys
                    print(
                        f"[build_raw_domain] feature mismatch (line {lineno} {source_name}): "
                        f"missing={sorted(missing)[:12]!s} (sample requires these in aggregate_chunk_from_hands output)",
                        file=sys.stderr,
                    )
                continue

            row: Dict[str, Any] = {k: float(feat[k]) for k in feature_names}
            row["label"] = None
            batch.append(row)
            accepted += 1

            if args.log_every and accepted % args.log_every == 0:
                print(f"[build_raw_domain] accepted_rows={accepted} (streaming...)")

            if len(batch) >= args.batch_size:
                flush()

        print(f"[build_raw_domain] done {source_name} accepted_total={accepted}")

    flush()
    if writer is not None:
        writer.close()
        writer = None

    if accepted == 0:
        print("[build_raw_domain] error: no rows written", file=sys.stderr)
        if out_path.is_file():
            out_path.unlink()
        return 1

    print(
        "[build_raw_domain] summary: "
        f"lines={total_lines} accepted={accepted} dup_hash={n_dup} "
        f"bad_json={n_bad_json} empty_chunk={total_skipped_empty} bad_agg={n_bad_chunk} "
        f"missing_hash_skipped={n_missing_hash} feat_mismatch={n_feat_mismatch}"
    )
    print(f"[build_raw_domain] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
