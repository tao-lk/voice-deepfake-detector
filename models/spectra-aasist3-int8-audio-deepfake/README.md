---
license: apache-2.0
base_model:
  - lab260/Spectra-AASIST3
tags:
  - onnx
  - audio
  - anti-spoofing
  - audio-deepfake-detection
  - speech
  - quantized
  - int8
---

# Spectra-AASIST3 INT8 — Academic Deployment Derivative

**Base model:** [`lab260/Spectra-AASIST3`](https://huggingface.co/lab260/Spectra-AASIST3)
**Original revision:** `bc0ded888080ddad493177bb53aa6f5b95219d7c`
**Original license:** Apache-2.0

This repository hosts a **dynamic INT8 ONNX derivative** of
`lab260/Spectra-AASIST3`, produced for a university Final Year Project
("Detecting AI-Generated and Cloned Voices: A Deep Learning System for
Robust Audio Deepfake Detection", GitHub:
[`munz-aftab/audio-deepfake-detection-fyp`](https://github.com/munz-aftab/audio-deepfake-detection-fyp)),
to make CPU-only browser-based deployment (Streamlit Community Cloud)
practical without a GPU.

## Derivation

- **No retraining. No fine-tuning.** This is a numerical transform of the
  original FP32 checkpoint's weights only.
- Produced with **ONNX Runtime dynamic INT8 quantization**
  (`onnxruntime.quantization.quantize_dynamic`, `op_types_to_quantize=['MatMul']`,
  `weight_type=QInt8`, `per_channel=False`) — primarily quantizing the
  wav2vec2 XLS-R-300m transformer's MatMul/linear-projection weights,
  which carry 95.4% of the original model's weight bytes. Convolutional
  layers (the RawNet2-style encoder, ~4% of weight bytes) are left in
  FP32.
- Generated from the original FP32 ONNX artifact
  (`spectra-aasist3.onnx`, SHA256
  `5f05c29a01ad80c702b32654db87c2aa6e467c11c67b6d47f2fac873f846cae9`),
  itself an export of `lab260/Spectra-AASIST3`'s `model.safetensors`.
- **CPU deployment target.** `onnxruntime` CPU execution provider only;
  never tested with or intended for CUDA/GPU.

## Architecture

wav2vec 2.0 XLS-R-300m self-supervised front-end → single-layer MLP bridge
→ KAN-enhanced AASIST (KAN-AASIST) back-end. See the base model card for
full architectural detail; this repository changes only the numerical
precision of the exported ONNX graph's weights, not the architecture.

## Input

- Mono audio, 16 kHz
- Pre-emphasis, coefficient 0.97, applied to the full waveform
  (`y[0] = x[0]`, `y[n] = x[n] - 0.97*x[n-1]` for `n ≥ 1`)
- Exactly 64,600 samples (~4.04s) required by the graph
- **Long clips**: deterministic first-64,600-sample window (author-
  compatible scoring convention)
- **Short clips**: tile-repeat (not zero-pad) until ≥64,600 samples, then
  truncated to exactly 64,600

## Output

- 2 logits, `wav → logits`
- **Class/logit index 1 = bona fide.** Higher bona-fide score/logit means
  more likely genuine speech. (Verified directly from the base model's
  own source code — see the base model's `model.py`,
  `SpectraAASIST3.classify()`.)

## Evaluation — FYP project-measured results on a fixed balanced In-the-Wild evaluation subset

**These are NOT the original model authors' benchmark results, and are
not claimed to represent universal real-world performance.** They were
measured by this FYP on a small (200-clip), balanced, deterministically
sampled subset of the `SpeechAntiSpoofingBenchmarks/InTheWild` dataset,
disjoint from a separate calibration subset used to select the operating
threshold. Full methodology:
[`docs/spectra_aasist3_evaluation.md`](https://github.com/munz-aftab/audio-deepfake-detection-fyp/blob/feat/spectra-aasist3-eval/docs/spectra_aasist3_evaluation.md)
and
[`docs/spectra_production_optimization.md`](https://github.com/munz-aftab/audio-deepfake-detection-fyp/blob/feat/spectra-production-optimization/docs/spectra_production_optimization.md)
in the project's GitHub repository.

| Metric | INT8 (this repo) | FP32 reference |
|---|---|---|
| EER | 3.0% | 2.0% |
| ROC-AUC | 0.9819 | 0.9766 |
| F1 | 0.9375 | 0.9326 |
| Bonafide FPR | 2.0% | 3.0% |
| Spoof FNR | 10.0% | 10.0% |

Do not treat 3.0% EER as a universal, dataset-independent performance
guarantee — it is a single-subset, project-measured number, reported for
transparency and reproducibility, not as a general benchmark claim.

## Limitations

- The base model, `lab260/Spectra-AASIST3`, is a **pre-release/unpublished**
  detector — there is no peer-reviewed paper for this exact architecture
  as of this writing.
- Domain shift (languages, recording conditions, unseen synthesis methods)
  remains possible; the evaluation above covers one real-world corpus at
  one point in time.
- **Not forensic evidence.** Not suitable as the sole basis for legal,
  security, or identity-verification decisions.
- Output scores are raw logits/softmax values, **not inherently calibrated
  probabilities** — a "0.9" score is not literally "90% likely genuine."
- Long-audio (multi-window) aggregation is an **application-level
  extension** built by this FYP, not part of the original model or its
  authors' evaluation methodology.

## Files

- `spectra-aasist3-int8-dynamic.onnx` — the quantized model
- `quantization_metadata.json` — full reproducibility metadata (base
  revision/hash, ONNX Runtime version, quantized operator counts,
  resulting hash/size, evaluation summary, date)

## Attribution

Base model and architecture: Kirill Borodin / `lab260`
(`lab260/Spectra-AASIST3`), Apache-2.0. Upstream SSL backbone:
`facebook/wav2vec2-xls-r-300m` (Meta AI / Facebook, Apache-2.0). This
derivative repository: produced as part of a university Final Year
Project; not affiliated with the original authors.
