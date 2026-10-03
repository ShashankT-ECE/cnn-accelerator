#!/usr/bin/env bash
# copied from v2/dpu/run_docker.sh @ 28dd2ad (V3: v3/ paths, v3 nets, guard = v3/scripts/vivado_guard.sh)
# DPU baseline: run one step inside the Vitis AI 2.5.0 CPU docker image (laptop).
#
#   v3/dpu/run_docker.sh arch             # arch.json from the prebuilt pynq-dpu 2.5 KV260 overlay
#                                         #   (needs fetch_overlay_ref.sh first)
#   v3/dpu/run_docker.sh quantize <net>   # vai_q_pytorch calib + test + deploy (export xmodel)
#   v3/dpu/run_docker.sh compile  <net>   # vai_c_xir for that arch.json + inspect_xmodel.py
#                                         #   (fails unless fingerprint == overlay's)
#   v3/dpu/run_docker.sh all              # arch + quantize + compile for $DPU_NETS
#                                         #   (default "resnet20_b mobilenet_s"), one container at a time
#   v3/dpu/run_docker.sh shell            # interactive shell (debug)
#
# Inputs: v3/dpu/build/<net>/{dpu_inputs.npz, fp32_state_dict.pt, export_info.json} (export_data.py).
# Mounts: repo read-only at /workspace/repo; v3/dpu/build read-write at /workspace/build.
# Runs as the calling uid:gid (no root-owned files in build/). Limits: --cpus 8 --memory 12g.
# Resource rule (v3/CLAUDE.md): the container waits for v3/scripts/vivado_guard.sh (no Vivado job
# running, >= 8 GB available); polled every 60 s up to GUARD_MAX_MIN (default 240) minutes.
set -euo pipefail

IMAGE="${VAI_IMAGE:-xilinx/vitis-ai-cpu:2.5.0}"
DPU_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$DPU_DIR/../.." && pwd)"
BUILD="${DPU_BUILD_DIR:-$DPU_DIR/build}"
ARCH="${DPU_ARCH_JSON:-/workspace/build/arch_kv260_pynqdpu25.json}"   # container path (overlay_arch.py)
GUARD="$REPO/v3/scripts/vivado_guard.sh"
GUARD_MAX_MIN="${GUARD_MAX_MIN:-240}"
NETS="${DPU_NETS:-resnet20_b mobilenet_s}"
mkdir -p "$BUILD/logs"

guard() {
  local i
  for ((i = 0; i <= GUARD_MAX_MIN; i++)); do
    if "$GUARD" > "$BUILD/logs/guard_last.txt" 2>&1; then tail -1 "$BUILD/logs/guard_last.txt"; return 0; fi
    echo "[guard] $(date +%T) waiting: $(tail -1 "$BUILD/logs/guard_last.txt")"
    sleep 60
  done
  echo "[guard] gave up after ${GUARD_MAX_MIN} min" >&2; exit 2
}

run() {  # run <log-name> <bash command inside the container>
  guard
  local log="$BUILD/logs/$1.log"
  echo "[run_docker] $IMAGE :: $2  (log $log)"
  docker run --rm --cpus 8 --memory 12g --memory-swap 12g \
    -u "$(id -u):$(id -g)" -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 -e REPO=/workspace/repo -e DPU_BUILD=/workspace/build \
    -v "$REPO":/workspace/repo:ro -v "$BUILD":/workspace/build \
    -w /workspace/build --entrypoint /bin/bash "$IMAGE" \
    -c "source /opt/vitis_ai/conda/etc/profile.d/conda.sh && conda activate vitis-ai-pytorch && $2" \
    2>&1 | tee "$log"
  return "${PIPESTATUS[0]}"
}

quantize() {
  local n="$1" py=/workspace/repo/v3/dpu/vai_quantize.py
  run "quantize_${n}_calib" "python $py --net $n --mode calib"
  run "quantize_${n}_test" "python $py --net $n --mode test"
  run "quantize_${n}_deploy" "python $py --net $n --mode deploy"
}

arch() {
  run arch "python /workspace/repo/v3/dpu/overlay_arch.py"
}

compile() {
  local n="$1" q
  q="$(cd "$BUILD/$n/quantized" && ls *_int.xmodel)"
  run "compile_${n}" "rm -rf /workspace/build/$n/compiled && \
    vai_c_xir -x /workspace/build/$n/quantized/$q -a $ARCH \
      -o /workspace/build/$n/compiled -n ${n}_kv260 && \
    python /workspace/repo/v3/dpu/inspect_xmodel.py --net $n --arch $ARCH \\
      --expect-overlay /workspace/build/overlay_arch_info.json"
}

case "${1:-}" in
  arch) arch ;;
  quantize) quantize "$2" ;;
  compile) compile "$2" ;;
  all) arch; for n in $NETS; do quantize "$n"; compile "$n"; done ;;
  shell)
    guard
    docker run --rm -it --cpus 8 --memory 12g -u "$(id -u):$(id -g)" -e HOME=/tmp \
      -v "$REPO":/workspace/repo:ro -v "$BUILD":/workspace/build -w /workspace/build \
      --entrypoint /bin/bash "$IMAGE" ;;
  *) sed -n '2,13p' "$0"; exit 1 ;;
esac
