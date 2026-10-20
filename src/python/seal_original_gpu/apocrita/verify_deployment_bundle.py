#!/usr/bin/env python3
"""Verify every immutable file in a SEAL original deployment bundle."""

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
    root = args.root.resolve()
    manifest = root / "DEPLOYMENT_MANIFEST.sha256"
    if not manifest.is_file():
        raise SystemExit("Missing deployment manifest: {}".format(manifest))

    checked = 0
    for line_number, raw_line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            expected, relative = raw_line.split("  ", 1)
        except ValueError as error:
            raise SystemExit("Invalid deployment manifest line {}".format(line_number)) from error
        target = (root / relative).resolve()
        try:
            target.relative_to(root)
        except ValueError as error:
            raise SystemExit("Unsafe deployment path: {}".format(relative)) from error
        if not target.is_file():
            raise SystemExit("Missing deployment file: {}".format(relative))
        observed = sha256(target)
        if observed != expected:
            raise SystemExit("Checksum mismatch for {}".format(relative))
        checked += 1
    if checked == 0:
        raise SystemExit("Deployment manifest is empty")
    source_commit = (root / "SOURCE_COMMIT.txt").read_text(encoding="utf-8").strip()
    if len(source_commit) != 40:
        raise SystemExit("SOURCE_COMMIT.txt does not contain a full Git commit")
    print("[OK] Verified {} deployment files at commit {}".format(checked, source_commit))


if __name__ == "__main__":
    main()
