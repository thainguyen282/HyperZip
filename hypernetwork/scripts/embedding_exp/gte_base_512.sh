#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$root"
exec "${HYPERZIP_PYTHON:-python}" -m hypernetwork.train -c experiments/embedding_context_3b/gte_base_512.yaml "$@"
