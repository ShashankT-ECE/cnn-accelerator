#!/usr/bin/env bash
# copied from v2/dpu/fetch_overlay_ref.sh @ 28dd2ad (V3: output v3/dpu/build/pynq_dpu_2.5/)
# Download the pynq-dpu 2.5 sdist (PyPI) and, from its .link files, the PREBUILT KV260 overlay
# bitstream + hardware description + the Xilinx-compiled KV260 example xmodel; md5 checked against
# the .link files. Output (gitignored): v3/dpu/build/pynq_dpu_2.5/. Used by overlay_arch.py and
# make_baseline_package.py (expected overlay md5s).
#   v3/dpu/fetch_overlay_ref.sh
set -euo pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/build/pynq_dpu_2.5"
SDIST_URL=https://files.pythonhosted.org/packages/fc/97/88a22921c490fbb4bb8c14516bc54725883f81008382e6183148b9801fa7/pynq_dpu-2.5.tar.gz
SDIST_SHA=29110dc73610ecae85249b5cf8a3ecb6c3791d98ce60cd04ae77c2b013486f43
mkdir -p "$D/kv260" && cd "$D"
[[ -f pynq_dpu-2.5.tar.gz ]] || curl -fsSL -o pynq_dpu-2.5.tar.gz "$SDIST_URL"
echo "$SDIST_SHA  pynq_dpu-2.5.tar.gz" | sha256sum -c -
tar xzf pynq_dpu-2.5.tar.gz
link() {  # link <.link file> <board> -> "url md5"
  python3 -c "import json,sys;d=json.load(open(sys.argv[1]))[sys.argv[2]];print(d['url'],d['md5sum'])" "$1" "$2"
}
for lf in pynq_dpu-2.5/pynq_dpu/dpu.hwh.link pynq_dpu-2.5/pynq_dpu/dpu.bit.link \
          pynq_dpu-2.5/pynq_dpu/notebooks/dpu_mnist_classifier.xmodel.link; do
  read -r url sum < <(link "$lf" KV260)
  f="kv260/${url##*filename=}"
  [[ -f "$f" ]] || curl -fsSL -o "$f" "$url"
  echo "$sum  $f" | md5sum -c -
done
ls -l kv260
