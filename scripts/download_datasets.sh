#!/usr/bin/env bash
# Download the three real datasets used by the reproduction pipeline.
#
# Layout produced (all under ./datasets/, git-ignored):
#   datasets/divine_pilot4/      DIVINE Pilot-4 farm IoT streams  (~5 MB, Zenodo)
#   datasets/weednet/            WeedNet Sequoia multispectral    (~0.2 GB, GitHub)
#   datasets/ip102/              IP102 insect-pest images         (~3.9 GB, Google Drive)
#
# Usage:
#   bash scripts/download_datasets.sh              # all three
#   bash scripts/download_datasets.sh divine       # one only (divine|weednet|ip102)
#
# Requirements: curl, git, and (for IP102) gdown  ->  pip install gdown
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="$HERE/datasets"
mkdir -p "$DATA"

want="${1:-all}"

dl_divine() {
  echo "==> DIVINE Pilot-4 (Zenodo record 11432589)"
  local out="$DATA/divine_pilot4"; mkdir -p "$out"
  local rec="https://zenodo.org/api/records/11432589"
  # Fetch the file list and download each CSV via its content URL.
  curl -fsSL "$rec" \
    | grep -oE '"self":\s*"[^"]*/files/[^"]*/content"' \
    | sed -E 's/.*"(https[^"]+)".*/\1/' \
    | while read -r url; do
        fn="$(basename "$(dirname "$url")")"
        echo "    $fn"
        curl -fsSL "$url" -o "$out/$fn"
      done
  echo "    done: $(ls "$out" | wc -l) files"
}

dl_weednet() {
  echo "==> WeedNet (github.com/inkyusa/weedNet, Sequoia multispectral)"
  if [ -d "$DATA/weednet/.git" ]; then
    git -C "$DATA/weednet" pull --ff-only
  else
    git clone --depth 1 https://github.com/inkyusa/weedNet.git "$DATA/weednet"
  fi
}

dl_ip102() {
  echo "==> IP102 (Google Drive, via gdown) -- ~3.9 GB"
  command -v gdown >/dev/null || { echo "ERROR: gdown not found. Run: pip install gdown"; return 1; }
  local out="$DATA/ip102"; mkdir -p "$out"
  # IP102 v1.1 shared Drive folder (classification + detection).
  gdown --folder "https://drive.google.com/drive/folders/1svFSy2Da3cVMvekBwe13mzyx38XZ9xWo" -O "$out"
  # Extract the classification tarball if present.
  local tar
  tar="$(find "$out" -name 'ip102_v1.1.tar' | head -1 || true)"
  [ -n "$tar" ] && tar -xf "$tar" -C "$(dirname "$tar")"
  echo "    done"
}

case "$want" in
  divine)  dl_divine ;;
  weednet) dl_weednet ;;
  ip102)   dl_ip102 ;;
  all)     dl_divine; dl_weednet; dl_ip102 ;;
  *) echo "usage: $0 [all|divine|weednet|ip102]"; exit 2 ;;
esac

echo "All requested datasets are under $DATA/"
