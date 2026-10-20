#!/usr/bin/env python3
"""Build a minimal checksummed SEAL original deployment bundle."""

import argparse
import datetime as dt
import hashlib
import json
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path


SEAL_GPU_FILES = (
    "__init__.py",
    "README.md",
    "generate_seal_original_manifest.py",
    "run_foodweb_seal_original_gpu.py",
    "seal_original_run_artifacts.py",
    "validate_seal_original_gpu_run.py",
    "tests/test_seal_original_gpu_pipeline.py",
)
APOCRITA_FILES = (
    "build_seal_original_apocrita_bundle.py",
    "create_seal_original_gpu_env.sh",
    "environment.apocrita.yml",
    "gpu_probe.py",
    "requirements-apocrita.txt",
    "seal_original_gpu_array.sbatch",
    "seal_original_gpushort.sbatch",
    "submit_seal_original_gpu.sh",
    "verify_deployment_bundle.py",
)
ORIGINAL_CORE_FILES = (
    "src/python/Main.py",
    "src/python/util_functions.py",
    "src/python/run_foodweb_seal.py",
)


def command_output(arguments, cwd):
    return subprocess.run(
        arguments, cwd=str(cwd), check=True, text=True, stdout=subprocess.PIPE
    ).stdout.strip()


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collect_files(repo_root):
    seal_root = repo_root / "src/python/seal_original_gpu"
    dgcnn_root = repo_root / "src/python/pytorch_DGCNN"
    files = [seal_root / relative for relative in SEAL_GPU_FILES]
    files.extend(seal_root / "apocrita" / relative for relative in APOCRITA_FILES)
    files.extend(repo_root / relative for relative in ORIGINAL_CORE_FILES)

    mat_root = repo_root / "src/matlab/data/foodwebs_mat"
    mat_files = sorted(mat_root.glob("*.mat"))
    if len(mat_files) != 290:
        raise RuntimeError("Expected 290 original MAT files, found {}".format(len(mat_files)))
    files.extend(mat_files)
    files.append(mat_root / "foodweb_metrics_ecosystem.csv")

    files.extend(sorted(dgcnn_root.glob("*.py")))
    files.append(dgcnn_root / "lib/Makefile")
    files.extend(sorted((dgcnn_root / "lib").glob("*.py")))
    files.extend(sorted((dgcnn_root / "lib/include").glob("*.h")))
    files.extend(sorted((dgcnn_root / "lib/src").rglob("*.cpp")))
    files.append(repo_root / "src/python/software/node2vec/src/node2vec.py")
    files.append(repo_root / "src/python/software/node2vec/LICENSE.md")

    missing = [path for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing bundle inputs: {}".format(missing))
    return sorted(set(files)), mat_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[4]
    commit = command_output(["git", "rev-parse", "HEAD"], repo_root)
    branch = command_output(["git", "branch", "--show-current"], repo_root)
    files, mat_files = collect_files(repo_root)
    relative_files = [path.relative_to(repo_root) for path in files]

    status = command_output(
        ["git", "status", "--porcelain", "--"] + [str(path) for path in relative_files],
        repo_root,
    )
    if status:
        raise SystemExit("Bundle inputs differ from commit {}:\n{}".format(commit, status))

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_name = "seal-original-apocrita-{}".format(commit[:12])
    archive = output_dir / "{}.tar.gz".format(bundle_name)
    checksum = output_dir / "{}.sha256".format(archive.name)
    if archive.exists() or checksum.exists():
        raise FileExistsError("Use an empty output directory: {}".format(output_dir))

    with tempfile.TemporaryDirectory(prefix="seal-original-bundle-") as temporary:
        bundle_root = Path(temporary) / bundle_name
        for source, relative in zip(files, relative_files):
            target = bundle_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        (bundle_root / "SOURCE_COMMIT.txt").write_text(commit + "\n", encoding="utf-8")
        metadata = {
            "schema": "seal-original-apocrita-bundle-v1",
            "source_commit": commit,
            "source_branch": branch,
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "mat_files": len(mat_files),
            "protocol": "unchanged historical SEAL original on CUDA; 10x290 ready",
        }
        (bundle_root / "BUNDLE_METADATA.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        entries = []
        for path in sorted(bundle_root.rglob("*")):
            if path.is_file() and path.name != "DEPLOYMENT_MANIFEST.sha256":
                entries.append("{}  {}".format(sha256(path), path.relative_to(bundle_root).as_posix()))
        (bundle_root / "DEPLOYMENT_MANIFEST.sha256").write_text(
            "\n".join(entries) + "\n", encoding="utf-8"
        )
        with tarfile.open(archive, "w:gz") as handle:
            handle.add(bundle_root, arcname=bundle_name)

    checksum.write_text("{}  {}\n".format(sha256(archive), archive.name), encoding="utf-8")
    print("[OK] Bundle: {}".format(archive))
    print("[OK] Checksum: {}".format(checksum))
    print("[OK] Files: {}; original MAT files: {}".format(len(relative_files), len(mat_files)))


if __name__ == "__main__":
    main()
