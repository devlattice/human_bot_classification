#!/usr/bin/env bash
# Deprecated: production bundle is v12. Forwards to deploy_prod.sh.
echo "[warn] deploy_v11_prod.sh is deprecated — use deploy_prod.sh (v12 bundle)" >&2
export BUNDLE_REL="${BUNDLE_REL:-workspace/model/artifacts/model_bundle_v12_prod}"
exec "$(dirname "$0")/deploy_prod.sh" "$@"
