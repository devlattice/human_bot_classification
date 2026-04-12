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
cd /root/Workspace/miner_7/poker_miner_7

python3 -c "
import hashlib
from pathlib import Path

def sha256_for_files(paths):
    digest = hashlib.sha256()
    for path in sorted((p.resolve() for p in paths), key=lambda p: str(p)):
        digest.update(str(path).encode('utf-8'))
        with path.open('rb') as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
    return digest.hexdigest()

print(sha256_for_files([Path('neurons/miner.py')]))
"
```

Set:

```bash
POKER44_MODEL_IMPLEMENTATION_SHA256="<python output>"
```

## 4) Quick checklist

- Update `.env` with all four values.
- Restart miner process so new manifest values are published.
- Check miner logs for manifest/compliance status.
