"""
Dry-run checks for the MuJoCo Batch launcher.

Runs launch_mujoco_batch.py --dry-run as a user would and checks the SubmitJob
payload it prints. Makes no AWS calls and needs no boto3:
  pytest training/scripts/test_launch_mujoco_batch.py
or, with no pytest installed (the fast lane):
  python3 training/scripts/test_launch_mujoco_batch.py
"""

import json
import os
import subprocess
import sys
from pathlib import Path

LAUNCHER = Path(__file__).parent / "launch_mujoco_batch.py"
MARKER = "[dry-run] Would call batch.submit_job with:"


def dry_run(*args, **env_overrides):
    env = {k: v for k, v in os.environ.items() if not k.startswith("MUJOCO_BATCH_")}
    env.pop("PROJECT_NAME", None)
    env.pop("ENVIRONMENT", None)
    env.update(env_overrides)
    result = subprocess.run(
        [sys.executable, str(LAUNCHER), "--dry-run", *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    out = result.stdout
    assert MARKER in out
    assert "[dry-run] No AWS calls made." in out
    body = out[out.index(MARKER) + len(MARKER):]
    payload, _ = json.JSONDecoder().raw_decode(body[body.index("{"):])
    return payload


def env_overrides(payload):
    return {e["name"]: e["value"] for e in payload["containerOverrides"]["environment"]}


def test_dry_run_defaults_target_terraform_names():
    payload = dry_run()

    # Names created by mujoco-on-aws/infra/batch.tf
    assert payload["jobQueue"] == "physical-ai-dev-mujoco-queue"
    assert payload["jobDefinition"] == "physical-ai-dev-mujoco-train"
    assert payload["jobName"].startswith("mujoco-rl-")
    assert env_overrides(payload) == {
        "ENV_NAME": "CartpoleBalance",
        "NUM_ENVS": "2048",
        "NUM_TIMESTEPS": "5000000",
        "SEED": "0",
    }
    # Single-node container job: no multi-node overrides
    assert "nodeOverrides" not in payload


def test_dry_run_passes_flags_through():
    payload = dry_run(
        "--env-name", "WalkerWalk",
        "--num-envs", "8",
        "--num-timesteps", "8000",
        "--seed", "7",
        "--job-queue", "my-queue",
        "--job-definition", "my-jobdef:3",
    )

    assert payload["jobQueue"] == "my-queue"
    assert payload["jobDefinition"] == "my-jobdef:3"
    assert env_overrides(payload) == {
        "ENV_NAME": "WalkerWalk",
        "NUM_ENVS": "8",
        "NUM_TIMESTEPS": "8000",
        "SEED": "7",
    }


def test_dry_run_reads_queue_and_job_definition_from_env():
    payload = dry_run(MUJOCO_BATCH_QUEUE="env-queue", MUJOCO_BATCH_JOB_DEF="env-jobdef")

    assert payload["jobQueue"] == "env-queue"
    assert payload["jobDefinition"] == "env-jobdef"


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"{len(tests)} passed")
