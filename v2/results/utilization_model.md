# PE-array utilization per layer (model)

Rendered by `v2/analysis/utilization.py` from `utilization_model.csv` (source=model; rows git_commit e9cbb886, git_dirty False).

Definitions: spatial = OC*OH*OW*K / (64*T*K); temporal = T*K / LAYER_CYC (MAC_ACTIVE/LAYER_CYC); total = spatial x temporal. All numbers are the analytical cycle model, not RTL or hardware measurements.

| net | layer | OC | OH x OW | K | T | LAYER_CYC | cycle share | eff_oc | eff_x | spatial | temporal | total | loss |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| lenet5 | conv1 | 6 | 28x28 | 25 | 112 | 2829 | 17.2% | 75.0% | 87.5% | 65.6% | 99.0% | 65.0% | x tail: OW=28 padded to 32; OC tail: OC=6 padded to 8; pipeline fill C_PIPE |
| lenet5 | conv3 | 16 | 10x10 | 150 | 40 | 6029 | 36.7% | 100.0% | 62.5% | 62.5% | 99.5% | 62.2% | x tail: OW=10 padded to 16; pipeline fill C_PIPE |
| lenet5 | conv5 | 120 | 1x1 | 400 | 15 | 6029 | 36.7% | 100.0% | 12.5% | 12.5% | 99.5% | 12.4% | 1x1 output: 1 of 8 rows active; pipeline fill C_PIPE |
| lenet5 | fc1 | 84 | 1x1 | 120 | 11 | 1349 | 8.2% | 95.5% | 12.5% | 11.9% | 97.9% | 11.7% | 1x1 output: 1 of 8 rows active; OC tail: OC=84 padded to 88; pipeline fill C_PIPE |
| lenet5 | fc2 | 10 | 1x1 | 84 | 2 | 197 | 1.2% | 62.5% | 12.5% | 7.8% | 85.3% | 6.7% | 1x1 output: 1 of 8 rows active; OC tail: OC=10 padded to 16; pipeline fill C_PIPE |
| lenet5 | **total** | | | | 180 | 16436 | | | | 40.0% | 99.1% | **39.6%** | TOTAL_CYC basis |
| cifar10 | conv1 | 32 | 28x28 | 75 | 448 | 33629 | 32.3% | 100.0% | 87.5% | 87.5% | 99.9% | 87.4% | x tail: OW=28 padded to 32; pipeline fill C_PIPE |
| cifar10 | conv2 | 32 | 10x10 | 800 | 80 | 64029 | 61.4% | 100.0% | 62.5% | 62.5% | 100.0% | 62.5% | x tail: OW=10 padded to 16; pipeline fill C_PIPE |
| cifar10 | conv3 | 64 | 1x1 | 800 | 8 | 6429 | 6.2% | 100.0% | 12.5% | 12.5% | 99.5% | 12.4% | 1x1 output: 1 of 8 rows active; pipeline fill C_PIPE |
| cifar10 | fc | 10 | 1x1 | 64 | 2 | 157 | 0.2% | 62.5% | 12.5% | 7.8% | 81.5% | 6.4% | 1x1 output: 1 of 8 rows active; OC tail: OC=10 padded to 16; pipeline fill C_PIPE |
| cifar10 | **total** | | | | 538 | 104247 | | | | 67.4% | 99.9% | **67.3%** | TOTAL_CYC basis |

## Why the 1x1-output layers are at most 1/8

Array rows are 8 consecutive output x positions of one output row and columns are 8 output channels (ARCH_SPEC Datapath). A layer whose output map is 1x1 has one output pixel per channel, so only row 0 carries a real pixel; the other 7 rows compute masked outputs. Its spatial utilization is therefore at most 12.5% (eff_x = 1/8), lower when OC is not a multiple of 8. Affected layers (derived from the shapes):

- lenet5 conv5: spatial 12.5% (exact 1/8), 36.7% of the net's cycles.
- lenet5 fc1: spatial 11.9% (exact 21/176), 8.2% of the net's cycles.
- lenet5 fc2: spatial 7.8% (exact 5/64), 1.2% of the net's cycles.
- cifar10 conv3: spatial 12.5% (exact 1/8), 6.2% of the net's cycles.
- cifar10 fc: spatial 7.8% (exact 5/64), 0.2% of the net's cycles.

## Where the array slots go (per net, share of 64 x TOTAL_CYC)

- lenet5: useful MACs 39.6%, x tail (incl. 1x1 outputs) 55.7%, OC tail 3.8%, pipeline fill + start 0.9%; 1x1-output layers take 46.1% of the cycles.
- cifar10: useful MACs 67.3%, x tail (incl. 1x1 outputs) 32.5%, OC tail 0.0%, pipeline fill + start 0.1%; 1x1-output layers take 6.3% of the cycles.

Temporal utilization is close to 1 because the per-layer overhead is the fixed C_PIPE fill/flush (no per-pass lead-in or drain, DECISIONS D7); the utilization loss is almost entirely spatial (tile padding).
