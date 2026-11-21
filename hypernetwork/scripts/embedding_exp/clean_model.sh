#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$root"
# Pass --step, --input and --output. The clean baseline has no hypernetwork.
exec "${HYPERZIP_PYTHON:-python}" main.py --model nemotron \
  --model_path nvidia/Nemotron-Labs-Diffusion-3B-Base \
  --revision 4bd78816ba988ca8b71b8f432b00b682572b236c \
  --trust_remote_code "$@"
