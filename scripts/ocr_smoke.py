"""Run PaddleOCR on generated slides and persist the exact output."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import paddle
import paddleocr

from evidenceagent_mm.perception import PaddleOCRAdapter


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+")
    parser.add_argument("--device", default="gpu")
    parser.add_argument("--detection-model", default="PP-OCRv5_mobile_det")
    parser.add_argument("--recognition-model", default="PP-OCRv5_mobile_rec")
    parser.add_argument("--output", default="results/gpu/ocr_smoke.json")
    args = parser.parse_args()
    started = time.perf_counter()
    adapter = PaddleOCRAdapter(
        lang="en",
        device=args.device,
        detection_model=args.detection_model,
        recognition_model=args.recognition_model,
    )
    atoms = []
    for index, image in enumerate(args.images):
        atoms.extend(adapter.extract(image, "synthetic-ocr-smoke", index * 7_000))
    result = {
        "backend": "PaddleOCR",
        "elapsed_seconds": time.perf_counter() - started,
        "paddle_version": paddle.__version__,
        "paddleocr_version": paddleocr.__version__,
        "paddle_device": paddle.device.get_device(),
        "device_requested": args.device,
        "cuda_compiled": paddle.device.is_compiled_with_cuda(),
        "pipeline_options": {
            "lang": "en",
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
            "text_detection_model_name": args.detection_model,
            "text_recognition_model_name": args.recognition_model,
        },
        "gpu": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            text=True,
        ).strip(),
        "atoms": [atom.model_dump(mode="json") for atom in atoms],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"atoms": len(atoms), "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
