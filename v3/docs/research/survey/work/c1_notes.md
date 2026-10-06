# Category 1 notes: FPGA CNN accelerators on Zynq UltraScale+ (KV260/K26, ZCU102, ZCU104, Ultra96)

Date: 2026-10-07. Output: `c1.csv` (13 rows, ranked best first).

## Method
- Started from `research/work/t5_findings.md`, `t5_related.csv`, `verify_report.csv`.
- Every DOI was re-checked at Crossref (title, first author, year all match for the 13 rows).
- Abstracts were read from OpenAlex records (the IEEE Xplore pages and PDFs returned
  empty pages or HTTP 418 to our fetches). Full text was read where an open copy was
  reachable: Wu 2019 (course-hosted copy of the IEEE version), DNNVM, FINN-R, Synetgy
  and Salami (arXiv). DNNVM, FINN-R and Synetgy were read at section or grep level only;
  `read_locator` says which parts.
- Basis count: 5 full text, 8 abstract.

## Search queries (web)
- "Kria KV260 CNN accelerator ResNet MobileNet FPGA IEEE 2024"
- "ZCU102 MobileNetV2 accelerator depthwise separable FPGA journal 2022"
- "ZCU104 ResNet-50 FPGA accelerator HLS INT8 throughput paper"
- "per-layer profiling DPUCZDX8G Kria KV260 MobileNet V1 V2 ResNet-50"
- "An FPGA-Based CNN Accelerator Integrating Depthwise Separable Convolution ZCU102"
- "FPGA CNN accelerator Ultra96 ZU3EG ResNet INT8 systolic array"
- Knapheide FPL 2020 MobileNetV2 (title search)
- "FPGA ResNet-18 ResNet-50 accelerator ZCU102 ... Zynq UltraScale+"
- Crossref title queries for Knapheide 2020 and RADAR 2025

## AMD/Xilinx DPU
No peer-reviewed DOI paper describes the DPUCZDX8G product IP itself. The DPU line is
covered by Wu 2019 (FPL, MobileNet DPU from the DeePhi/Xilinx team), DNNVM (TCAD 2020,
the compiler for the DPU-style ISA), and by DPU evaluation papers: Hamanaka 2023 (K26),
Magalhães 2023 (KV260/ZCU104) and Salami 2020 (ZCU102). The vendor model zoo is not peer
reviewed, so it is not a row.

## Considered and dropped
- tong2024feather (ISCA 2024, ZCU104): the main contribution is reconfigurable dataflow,
  which is owned by another category. Spare if that category drops it.
- zhang2022wsqaddernet (ICCAD 2022, KV260 ResNet20 INT8): the main contribution is
  DSP-LUT co-packing and AdderNet, owned by the DSP-packing/quantization category. It is
  also closed access and was not read.
- sun2022filmqnn, zhang2021fracbnn, saha2026potacc: quantization-method papers, owned elsewhere.
- zhao2021dualopu: its own design is on a Kintex-7 (XC7K325T), not ZU+.
- lu2017evaluating (10.1109/FCCM.2017.64, Winograd on ZCU102, AlexNet/VGG16): DOI
  verified and abstract read. Spare 1. It was ranked lower because it does not cover
  ResNet/MobileNet and is a fast-algorithm study.
- bjerge2021scalable (10.1016/j.micpro.2021.104363, HLS CNN accelerator on Ultra96):
  DOI verified, arXiv read at grep level. Spare 2. It was ranked lower because it evaluates
  VGG16 only.
- costa2026throughput (JRTIP 2026, KV260 DPU multithreading): DOI verified, but the full
  text and abstract are behind a Springer login. Not used.
- knapheide2020 (10.1109/fpl50879.2020.00053, MobileNetV2 1050 fps, 34 W): the abstract
  does not name the device, and the 34 W suggests it is not ZU+. Dropped.
- huang2023 (Electronics 12(7):1571, depthwise-separable accelerator): Intel Arria 10. Dropped.
- "Measuring What Actually Matters: per-layer DPU profiling on KV260": Medium/blog only,
  not peer reviewed. Dropped.
- NorCAS 2024 KV260 vs Jetson Nano (search snippet only): no DOI was checked. Lead only.
- Minnella 2023 arXiv (nn2fpga preprint): superseded by the TCAD version.
- Tensil ResNet-20 on Ultra96: blog only.

## Caveats
- bosio2025nn2fpga: the KV260 ResNet-20 numbers (7601 FPS, 0.318 ms, 6.10 W, 636 DSP,
  250 MHz) come from the earlier t5 full-text pass. They were not re-read here, so they are
  only in `notes`/`relevance`, not in the annotation columns.
- park2025radar is new and from a smaller venue (JSTS). Only the abstract was read.
- dong2021hao (NAS + quantization co-design) may overlap with another category. Spares
  are listed above.
