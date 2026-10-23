#!/usr/bin/env python3
"""Prove real CUDA execution and native DGCNN compatibility before a task."""

import ctypes
import hashlib
import json
import os
import platform
from pathlib import Path

import torch


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = torch.device("cuda:0")
    left = torch.randn(2048, 2048, device=device, requires_grad=True)
    right = torch.randn(2048, 2048, device=device)
    loss = (left @ right).square().mean()
    loss.backward()
    library = Path(__file__).resolve().parents[2] / "pytorch_DGCNN/lib/build/dll/libgnn.so"
    if not library.is_file():
        raise FileNotFoundError(library)
    ctypes.CDLL(str(library))
    digest = hashlib.sha256(library.read_bytes()).hexdigest()
    print(json.dumps({
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_memory": torch.cuda.get_device_properties(0).total_memory,
        "loss": float(loss.detach().cpu()),
        "native_library": str(library),
        "native_library_sha256": digest,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

