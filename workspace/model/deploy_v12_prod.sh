#!/usr/bin/env bash
# Deploy model_bundle_v12_prod (wrapper).
exec "$(dirname "$0")/deploy_prod.sh" "$@"
