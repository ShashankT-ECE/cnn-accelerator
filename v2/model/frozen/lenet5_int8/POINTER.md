# LeNet-5 INT8 frozen reference — pointer (not a copy)

The V2 LeNet-5 INT8 reference parameters are the legacy frozen L2 artifact, used in place:

- File: `data/lenet5_int8/quant_params.npz` (tracked in git; legacy, read-only)
- SHA256: `3d39eb008d0e20f934c3d70288d6142d50a55e933129aa285430a06407a6ff84`
  (as listed in `data/lenet5_int8/SHA256SUMS`)
- Generator: `python/export_lenet5_int8.py` (calibration: MNIST train[0:1024], no RNG)
- Provenance: `data/lenet5_int8/manifest.json`, `data/lenet5_int8/L2_FROZEN_MANIFEST.json`
- Keys: `S_input`, `{conv1,conv3,conv5,fc1,fc2}_{q_w,q_b,S_w,S_a}`, and
  `{conv1,conv3,conv5,fc1}_{S_out,M}` (final layer `fc2` has no S_out/M)

Verified by `v2/model/tests/test_lenet_pointer.py` (hash vs `data/lenet5_int8/SHA256SUMS`
and vs the value above).
