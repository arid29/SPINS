#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
revision=8829d14f814fccdaf840b7b0f3021a616583c0a1
destination=third_party/diff-gaussian-rasterization
if [[ ! -d "$destination/.git" ]]; then
  git clone https://github.com/ashawkey/diff-gaussian-rasterization.git "$destination"
fi
git -C "$destination" checkout "$revision"
git -C "$destination" submodule update --init --recursive
"${SPINS_PYTHON:-python}" -m pip install --no-build-isolation "$destination"
