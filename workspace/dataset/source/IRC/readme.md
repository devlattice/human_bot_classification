# IRC Build Commands

Run from repo root:

```bash
cd /home/dr/Workspace/Poker44-subnet
```

## 1) Quick smoke test (small sample)

```bash
python3 workspace/dataset/source/IRC/build.py --sample 5000
```

## 2) Full build (download + extract + parse)

```bash
python3 workspace/dataset/source/IRC/build.py
```

## 3) Resume-style run (skip network and extraction)

```bash
python3 workspace/dataset/source/IRC/build.py --skip-download --skip-extract
```

## 4) Tune log frequency in shell

```bash
python3 workspace/dataset/source/IRC/build.py \
  --log-every-lines 100000 \
  --log-every-hands 5000
```

## 5) Custom output paths

```bash
python3 workspace/dataset/source/IRC/build.py \
  --out-accepted workspace/dataset/source/IRC/poker_hands_irc_normalized.jsonl \
  --out-rejects workspace/dataset/source/IRC/poker_hands_irc_rejects.jsonl \
  --out-summary workspace/dataset/source/IRC/qc_summary.json
```

## 6) Check resulting file sizes

```bash
ls -lh workspace/dataset/source/IRC/*.json*
```

## 7) Peek accepted / rejected records

```bash
python3 - <<'PY'
import json
from itertools import islice

paths = [
    "workspace/dataset/source/IRC/poker_hands_irc_normalized.jsonl",
    "workspace/dataset/source/IRC/poker_hands_irc_rejects.jsonl",
]
for p in paths:
    print(f"\n== {p} ==")
    with open(p, "r", encoding="utf-8") as f:
        for line in islice(f, 3):
            obj = json.loads(line)
            print(obj)
PY
```

## 8) Summarize reject reasons from QC summary

```bash
python3 - <<'PY'
import json
p = "workspace/dataset/source/IRC/qc_summary.json"
with open(p, "r", encoding="utf-8") as f:
    s = json.load(f)
print("global_counts:", s.get("global_counts", {}))
print("top_reject_reasons:")
for k, v in sorted(s.get("reject_reason_counts", {}).items(), key=lambda kv: kv[1], reverse=True)[:20]:
    print(f"  {k}: {v}")
PY
```

## 9) Disk space checks (recommended before full build)

```bash
df -h .
du -sh workspace/dataset/source/IRC
```

## 10) Clean temp sqlite files if a run is interrupted

```bash
rm -f workspace/dataset/source/IRC/_tmp_sqlite/*.sqlite
```

## Notes

- Source archive is about 1 GB compressed; keep extra free space for extraction and outputs.
- This pipeline enforces schema/QC but does **not** guarantee all hands are human-labeled.
