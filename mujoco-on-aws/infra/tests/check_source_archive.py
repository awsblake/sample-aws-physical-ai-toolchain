"""
Zip-content check for the MuJoCo CodeBuild source archive.

codebuild_source.tftest.hcl can only see the archive_file arguments and the
source_dir on disk; Terraform has no function that reads a zip. This runs the
mocked-provider terraform test, which builds the real archive at plan time, then
opens that zip and checks it holds the Docker build context and none of the
secrets or Terraform state the excludes are meant to keep out:
  python3 mujoco-on-aws/infra/tests/check_source_archive.py
TERRAFORM names the binary (default: terraform on PATH).
"""

import fnmatch
import os
import subprocess
import sys
import zipfile
from pathlib import Path

INFRA = Path(__file__).resolve().parent.parent
ARCHIVE = INFRA / ".terraform" / "mujoco-source.zip"

REQUIRED = [
    "containers/mujoco/Dockerfile",
    "containers/mujoco/buildspec.yml",
    "containers/mujoco/test-image.sh",
    "containers/mujoco/train_entrypoint.py",
    "containers/mujoco/batch-train-entrypoint.sh",
]

# Matched against every path component, so a hit at any depth fails.
FORBIDDEN_PARTS = [".git", ".claude", ".beads", ".kiro", ".vscode", ".terraform", "node_modules", ".venv", "venv", "__pycache__"]
FORBIDDEN_NAMES = [".env", "*.env.local", "osmo_values.yaml", "*.tfstate", "*.tfstate.*", "*.tfvars", "*.pyc", ".DS_Store", "CLAUDE.md"]


def build_archive():
    terraform = os.environ.get("TERRAFORM", "terraform")
    for args in (["init", "-backend=false", "-input=false"], ["test"]):
        result = subprocess.run([terraform, *args], cwd=INFRA, capture_output=True, text=True)
        if result.returncode != 0:
            sys.stdout.write(result.stdout)
            sys.stderr.write(result.stderr)
            raise SystemExit(f"terraform {args[0]} failed ({result.returncode})")


def forbidden(name):
    parts = name.rstrip("/").split("/")
    if parts[0] == "plans" or name.startswith(("data/", "training/data/", "cdk/cdk.out", "cdk/cdk.context.json")):
        return True
    if any(p in FORBIDDEN_PARTS for p in parts):
        return True
    return any(fnmatch.fnmatch(parts[-1], pat) for pat in FORBIDDEN_NAMES)


def check(names):
    errors = [f"missing {f}" for f in REQUIRED if f not in names]
    errors += [f"must not ship {n}" for n in sorted(names) if forbidden(n)]
    return errors


def test_forbidden_matches_nested_secrets():
    assert forbidden("notebooks/lab1/.env")
    assert forbidden("deploy/osmo_values.yaml")
    assert forbidden("mujoco-on-aws/infra/terraform.tfstate.backup")
    assert forbidden("cdk/node_modules/x/index.js")
    assert not forbidden("containers/mujoco/Dockerfile")
    assert not forbidden("training/scripts/launch_mujoco_batch.py")


def test_archive_contents():
    build_archive()
    with zipfile.ZipFile(ARCHIVE) as zf:
        names = set(zf.namelist())
    errors = check(names)
    assert not errors, "\n".join(errors)


if __name__ == "__main__":
    test_forbidden_matches_nested_secrets()
    test_archive_contents()
    print("ok: source archive holds the MuJoCo build context and no excluded files")
