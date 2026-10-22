#!/usr/bin/env python3
"""Self-contained script to reproduce spectra-aasist3-int8-dynamic.onnx
from the original lab260/Spectra-AASIST3 FP32 ONNX export.

No project-specific imports -- only huggingface_hub, onnxruntime, and
hashlib. CPU only; does not use CUDA/TensorRT/fairseq.

Usage:
    pip install "huggingface_hub>=1.20" "onnxruntime>=1.20,<2" "onnx>=1.17,<2"
    python reproduce_quantization.py
"""
from __future__ import annotations

import hashlib
import os

BASE_REPO = "lab260/Spectra-AASIST3"
BASE_REVISION = "bc0ded888080ddad493177bb53aa6f5b95219d7c"
BASE_ONNX_FILENAME = "spectra-aasist3.onnx"
EXPECTED_BASE_SHA256 = "5f05c29a01ad80c702b32654db87c2aa6e467c11c67b6d47f2fac873f846cae9"
EXPECTED_INT8_SHA256 = "444f832d306a2be4f823119f84e698e8821db6a1aab248593d4b05b7a9a48108"


def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    from huggingface_hub import hf_hub_download
    from onnxruntime.quantization import QuantType, quantize_dynamic

    print(f"Downloading {BASE_REPO}@{BASE_REVISION} :: {BASE_ONNX_FILENAME} ...")
    fp32_path = hf_hub_download(repo_id=BASE_REPO, filename=BASE_ONNX_FILENAME, revision=BASE_REVISION)

    actual_base_sha256 = sha256_of_file(fp32_path)
    if actual_base_sha256 != EXPECTED_BASE_SHA256:
        raise RuntimeError(
            f"Base FP32 artifact SHA256 mismatch: expected {EXPECTED_BASE_SHA256}, "
            f"got {actual_base_sha256}. The upstream repository may have changed -- "
            "do not proceed without verifying."
        )
    print("Base FP32 SHA256 verified.")

    output_path = os.path.join(os.path.dirname(fp32_path), "spectra-aasist3-int8-dynamic.onnx")
    print("Quantizing (dynamic INT8, MatMul only) ...")
    quantize_dynamic(
        model_input=fp32_path,
        model_output=output_path,
        op_types_to_quantize=["MatMul"],
        weight_type=QuantType.QInt8,
        per_channel=False,
    )

    actual_int8_sha256 = sha256_of_file(output_path)
    print(f"Wrote {output_path}")
    print(f"Size: {os.path.getsize(output_path)} bytes")
    print(f"SHA256: {actual_int8_sha256}")
    if actual_int8_sha256 == EXPECTED_INT8_SHA256:
        print("Matches the published artifact's SHA256 exactly -- fully reproducible.")
    else:
        print(
            "NOTE: SHA256 differs from the published artifact "
            f"({EXPECTED_INT8_SHA256}). This can happen with different "
            "onnxruntime/onnx versions producing a byte-different but "
            "numerically-equivalent quantized graph -- verify numerically "
            "(see docs/spectra_production_optimization.md parity methodology) "
            "rather than assuming this indicates an error."
        )


if __name__ == "__main__":
    main()
