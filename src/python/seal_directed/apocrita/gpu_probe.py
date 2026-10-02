#!/usr/bin/env python3
"""Fail-fast CUDA and native-library probe for the Apocrita smoke job."""

import json
import os
import platform
import sys
import ctypes
from pathlib import Path

import torch


def main():
    if not torch.cuda.is_available():
        raise SystemExit("torch.cuda.is_available() is False")
    device = torch.device("cuda:0")
    left = torch.randn(2048, 2048, device=device, requires_grad=True)
    right = torch.randn(2048, 2048, device=device)
    loss = (left @ right).square().mean()
    loss.backward()
    torch.cuda.synchronize()

    script_dir = Path(__file__).resolve().parent.parent
    library = script_dir.parent / "pytorch_DGCNN" / "lib" / "build" / "dll" / "libgnn.so"
    with open(library, "rb") as handle:
        if handle.read(4) != b"\x7fELF":
            raise SystemExit("DGCNN native library is not ELF: {}".format(library))
    ctypes.CDLL(str(library))

    print(json.dumps({
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_memory": torch.cuda.get_device_properties(0).total_memory,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "loss": float(loss.detach().cpu()),
        "native_library": str(library),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
