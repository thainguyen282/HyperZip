#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$root"
exec "${HYPERZIP_PYTHON:-python}" -m hypernetwork.train -c hypernetwork/scripts/embedding_exp/gte_small_linear_192.yaml "$@"
