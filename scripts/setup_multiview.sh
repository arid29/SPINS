#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p third_party
if [ ! -d third_party/MV-Adapter/.git ]; then
    git clone https://github.com/huanngzh/MV-Adapter.git third_party/MV-Adapter
    git -C third_party/MV-Adapter checkout 4277e0018232bac82bb2c103caf0893cedb711be
fi
if [ ! -d .venv-mvadapter ]; then
    "${PYTHON:-python3}" -m venv --system-site-packages .venv-mvadapter
fi
.venv-mvadapter/bin/python -m pip install -r requirements-multiview.txt
.venv-mvadapter/bin/python scripts/prepare_multiview_weights.py
echo 'Use .venv-mvadapter/bin/python for multiview training and inference.'
