#!/usr/bin/env python3
"""Verify every checksummed file in an extracted deployment bundle."""

import argparse
import hashlib
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    manifest = args.root / "DEPLOYMENT_MANIFEST.sha256"
    lines = [line.strip() for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    for line in lines:
        expected, relative = line.split("  ", 1)
        path = args.root / relative
        if not path.is_file() or sha256(path) != expected:
            raise ValueError("deployment checksum mismatch: {}".format(path))
    commit = (args.root / "SOURCE_COMMIT.txt").read_text(encoding="utf-8").strip()
    print("[OK] Verified {} deployment files at commit {}".format(len(lines), commit))


if __name__ == "__main__":
    main()

