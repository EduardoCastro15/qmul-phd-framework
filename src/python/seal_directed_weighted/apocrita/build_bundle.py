#!/usr/bin/env python3
"""Build a minimal checksummed SEAL-directed-weighted Apocrita bundle."""

import argparse
import datetime as dt
import hashlib
import json
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path


PACKAGE_FILES = (
    "__init__.py", "README.md", "edge_schema.py", "build_weighted_mats.py",
    "run_artifacts.py", "split_bundles.py", "generate_manifest.py", "subgraphs.py",
    "models.py", "train_one.py", "run_campaign.py", "validate_run.py",
    "summarize_descriptive.py", "summarize_resources.py", "tests/test_weighted_pipeline.py",
)
APOCRITA_FILES = (
    "build_bundle.py", "create_gpu_env.sh", "environment.apocrita.yml", "gpu_probe.py",
    "requirements-apocrita.txt", "seal_weighted_gpu_array.sbatch", "submit_gpu.sh",
    "verify_bundle.py",
)


def output(command, cwd):
    return subprocess.run(command, cwd=cwd, check=True, text=True, stdout=subprocess.PIPE).stdout.strip()


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collect(repo):
    root = repo / "src/python/seal_directed_weighted"
    files = [root / name for name in PACKAGE_FILES]
    files.extend(root / "apocrita" / name for name in APOCRITA_FILES)
    mats = sorted((root / "data/foodwebs_mat_seal_weighted").glob("*.mat"))
    if len(mats) != 290:
        raise RuntimeError("expected 290 weighted MAT files, found {}".format(len(mats)))
    files.extend(mats)
    files.extend([
        root / "data/foodwebs_mat_seal_weighted/EDGE_SCHEMA.json",
        root / "data/foodwebs_mat_seal_weighted/weighted_mat_build_audit.csv",
        repo / "src/matlab/data/foodwebs_mat/foodweb_metrics_ecosystem.csv",
    ])
    backend = repo / "src/python/pytorch_DGCNN"
    files.extend(sorted(backend.glob("*.py")))
    files.append(backend / "lib/Makefile")
    files.extend(sorted((backend / "lib").glob("*.py")))
    files.extend(sorted((backend / "lib/include").glob("*.h")))
    files.extend(sorted((backend / "lib/src").rglob("*.cpp")))
    missing = [path for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    return sorted(set(files)), mats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[4]
    commit = output(["git", "rev-parse", "HEAD"], repo)
    branch = output(["git", "branch", "--show-current"], repo)
    files, mats = collect(repo)
    relatives = [path.relative_to(repo) for path in files]
    status = output(["git", "status", "--porcelain", "--"] + [str(path) for path in relatives], repo)
    if status:
        raise SystemExit("Bundle inputs differ from commit {}:\n{}".format(commit, status))
    destination = args.output_dir.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    name = "seal-directed-weighted-apocrita-{}".format(commit[:12])
    archive = destination / "{}.tar.gz".format(name)
    checksum = destination / "{}.sha256".format(archive.name)
    if archive.exists() or checksum.exists():
        raise FileExistsError(destination)
    with tempfile.TemporaryDirectory(prefix="seal-weighted-bundle-") as temporary:
        root = Path(temporary) / name
        for source, relative in zip(files, relatives):
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        (root / "SOURCE_COMMIT.txt").write_text(commit + "\n", encoding="utf-8")
        (root / "BUNDLE_METADATA.json").write_text(json.dumps({
            "schema": "seal-directed-weighted-apocrita-bundle-v1",
            "source_commit": commit,
            "source_branch": branch,
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "weighted_mat_files": len(mats),
            "variants": ["legacy_weak_node", "directed_node", "directed_scalar_gate", "directed_ecc"],
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        entries = []
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.name != "DEPLOYMENT_MANIFEST.sha256":
                entries.append("{}  {}".format(sha256(path), path.relative_to(root).as_posix()))
        (root / "DEPLOYMENT_MANIFEST.sha256").write_text("\n".join(entries) + "\n", encoding="utf-8")
        with tarfile.open(archive, "w:gz") as handle:
            handle.add(root, arcname=name)
    checksum.write_text("{}  {}\n".format(sha256(archive), archive.name), encoding="utf-8")
    print("[OK] Bundle: {}".format(archive))
    print("[OK] Checksum: {}".format(checksum))
    print("[OK] Files: {}; weighted MAT files: {}".format(len(relatives), len(mats)))


if __name__ == "__main__":
    main()
