#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$root"
exec "${HYPERZIP_PYTHON:-python}" -m hypernetwork.train -c hypernetwork/scripts/embedding_exp/gte_qwen2_1_5b_1534.yaml "$@"
