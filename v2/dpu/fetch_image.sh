#!/usr/bin/env bash
# Resumable fallback for `docker pull xilinx/vitis-ai-cpu:2.5.0` on slow/stalling links:
# downloads manifest + config + layer blobs from Docker Hub with curl (resume -C -, restart on
# stall), verifies every sha256, writes an OCI image layout and `docker load`s it.
#   v2/dpu/fetch_image.sh            (output: v2/dpu/build/image/, gitignored; ~6.4 GB)
set -euo pipefail
REPO=xilinx/vitis-ai-cpu; TAG=2.5.0
OUT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/build/image"
B="$OUT/blobs/sha256"; mkdir -p "$B"
tok() { curl -fsS "https://auth.docker.io/token?service=registry.docker.io&scope=repository:$REPO:pull" \
        | python3 -c "import sys,json;print(json.load(sys.stdin)['token'])"; }
MT=application/vnd.docker.distribution.manifest.v2+json
curl -fsS -H "Authorization: Bearer $(tok)" -H "Accept: $MT" \
  "https://registry-1.docker.io/v2/$REPO/manifests/$TAG" -o "$OUT/manifest.raw"
MD=$(sha256sum "$OUT/manifest.raw" | cut -d' ' -f1); cp "$OUT/manifest.raw" "$B/$MD"
echo "manifest sha256:$MD"
python3 -c "
import json;m=json.load(open('$OUT/manifest.raw'))
for d in [m['config']]+m['layers']: print(d['digest'][7:], d['size'])" > "$OUT/blobs.txt"
while read -r h size; do
  f="$B/$h"
  for try in $(seq 1 200); do
    have=$(stat -c %s "$f" 2>/dev/null || echo 0)
    [[ "$have" -ge "$size" ]] && break
    echo "$(date +%T) $h: $have/$size (try $try)"
    curl -sS -L -C - --speed-limit 20000 --speed-time 60 --retry 3 \
      -H "Authorization: Bearer $(tok)" "https://registry-1.docker.io/v2/$REPO/blobs/sha256:$h" \
      -o "$f" || true
  done
  got=$(sha256sum "$f" | cut -d' ' -f1)
  [[ "$got" == "$h" ]] || { echo "SHA MISMATCH $h (got $got)"; rm -f "$f"; exit 1; }
  echo "ok $h ($size B)"
done < "$OUT/blobs.txt"
printf '{"imageLayoutVersion":"1.0.0"}' > "$OUT/oci-layout"
printf '{"schemaVersion":2,"manifests":[{"mediaType":"%s","digest":"sha256:%s","size":%s,"annotations":{"io.containerd.image.name":"docker.io/%s:%s","org.opencontainers.image.ref.name":"%s"}}]}' \
  "$MT" "$MD" "$(stat -c %s "$OUT/manifest.raw")" "$REPO" "$TAG" "$TAG" > "$OUT/index.json"
echo "all blobs verified; loading"
tar -C "$OUT" -cf - oci-layout index.json blobs | docker load
docker image inspect "$REPO:$TAG" --format '{{.Id}} {{.Size}}'
