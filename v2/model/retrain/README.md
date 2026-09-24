# retrain — V2 Step 2.1b: CIFAR-10 retrain r2 (same architecture, NOT yet adopted)

The reference of record is still `v2/model/frozen/cifar10_int8/` (Step 2.1).
r2 is a candidate only; adoption is a separate decision.

Network, preprocessing and INT8 pipeline are the legacy ones, imported unchanged
(`cifar10.model.Cifar10Net`, `cifar10.preprocess.make_transform`,
`cifar10.int8_model.calibrate`, calibration on train[0:1024]). Epoch selection used
val = train[45000:50000] only; the test set was evaluated once, by `check_r2.py`.

Reproduce (repo root, `.venv`):

    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python v2/model/retrain/train_cifar10_r2.py
    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python v2/model/freeze_cifar10_int8.py \
        --ckpt v2/model/retrain/cifar10_fp32_r2.pt --out v2/model/frozen/cifar10_int8_r2
    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python v2/model/retrain/check_r2.py

Artifacts: `cifar10_fp32_r2.pt` (+ `.sha256`, `_meta.json`), `v2/model/frozen/cifar10_int8_r2/`
(`quant_params.npz`, `manifest.json`, `SHA256SUMS`, `hw_requant.npz` at B=32).

Results (label: model): `v2/results/cifar10_retrain_log.csv` (per epoch),
`cifar10_r2_accuracy.csv`, `requant_equivalence_r2.csv`, `final_layer_check_r2.csv`,
`cifar10_r2_summary.csv` (old vs new + acceptance-rule recommendation).
