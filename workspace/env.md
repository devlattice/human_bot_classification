# Manifest Value Setup

Run these commands from the repo root (`/home/dr/Workspace/Poker44-subnet`) to compute values for `.env`.

## 1) Repo commit SHA

```bash
git rev-parse HEAD
```

Set:

```bash
POKER44_MODEL_REPO_COMMIT="<40-char output>"
```

## 2) Model artifact SHA256

```bash
sha256sum "/home/dr/Workspace/Poker44-subnet/workspace/model/artifacts/lgbm_weak_ssl_ultralow_ssl_weak_mcp055/lgbm_b_classifier_calibrated.joblib"
```

Set:

```bash
POKER44_MODEL_ARTIFACT_SHA256="<first column output>"
```

Also set:

```bash
POKER44_MODEL_ARTIFACT_URL="<public URL for that joblib, or empty>"
```

## 3) Implementation SHA256 (inference code digest)

```bash
python - <<'PY'
import hashlib
from pathlib import Path

root = Path("/home/dr/Workspace/Poker44-subnet")
files = [
    root / "neurons" / "miner.py",
    root / "poker44" / "validator" / "chunk_features.py",
    root / "poker44" / "utils" / "model_manifest.py",
]
h = hashlib.sha256()
for p in files:
    b = p.read_bytes()
    h.update(str(p.relative_to(root)).encode() + b"\n")
    h.update(hashlib.sha256(b).hexdigest().encode() + b"\n")
print(h.hexdigest())
PY
```

Set:

```bash
POKER44_MODEL_IMPLEMENTATION_SHA256="<python output>"
```

## 4) Quick checklist

- Update `.env` with all four values.
- Restart miner process so new manifest values are published.
- Check miner logs for manifest/compliance status.
