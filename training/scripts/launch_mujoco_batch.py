"""
AWS Batch MuJoCo RL launcher.

Submits a single-node, single-GPU MuJoCo Playground training job (Brax PPO on
MJX) to the AWS Batch queue from mujoco-on-aws/infra/batch.tf. The job
definition runs containers/mujoco/batch-train; this script only overrides the
training env vars, so the image, GPU and CHECKPOINT_BUCKET come from the job
definition.

Queue/jobdef default to the names mujoco-on-aws/infra creates; pass the
Terraform outputs to override them.

Usage:
    # Preview the job (no AWS calls):
    python training/scripts/launch_mujoco_batch.py --dry-run

    # Smoke test on the default queue and job definition:
    python training/scripts/launch_mujoco_batch.py \
        --env-name CartpoleBalance --num-envs 2048 --num-timesteps 5000000

    # Use the queue and job definition from Terraform outputs:
    python training/scripts/launch_mujoco_batch.py \
        --env-name CartpoleBalance \
        --job-queue "$(terraform -chdir=mujoco-on-aws/infra output -raw batch_job_queue)" \
        --job-definition "$(terraform -chdir=mujoco-on-aws/infra output -raw batch_job_definition)"

NOTE: The MuJoCo component is in Preview: this path is not yet validated
end to end on hardware.
"""

import argparse
import json
import os
import time

# Matches the aws_region default in mujoco-on-aws/infra/variables.tf
REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
PROJECT_NAME = os.environ.get("PROJECT_NAME", "physical-ai")
ENVIRONMENT = os.environ.get("ENVIRONMENT", "dev")


def _default_queue() -> str:
    return os.environ.get("MUJOCO_BATCH_QUEUE", f"{PROJECT_NAME}-{ENVIRONMENT}-mujoco-queue")


def _default_job_def() -> str:
    return os.environ.get("MUJOCO_BATCH_JOB_DEF", f"{PROJECT_NAME}-{ENVIRONMENT}-mujoco-train")


def build_job_request(
    env_name: str,
    num_envs: int,
    num_timesteps: int,
    seed: int,
    job_queue: str,
    job_definition: str,
) -> dict:
    """Returns the batch.submit_job kwargs for one training job."""
    return {
        "jobName": f"mujoco-rl-{int(time.time())}",
        "jobQueue": job_queue,
        "jobDefinition": job_definition,
        # Env vars read by containers/mujoco/train_entrypoint.py. The job
        # definition sets CHECKPOINT_BUCKET; CHECKPOINT_PREFIX stays unset so
        # each job writes to checkpoints/<job_id>.
        "containerOverrides": {
            "environment": [
                {"name": "ENV_NAME", "value": env_name},
                {"name": "NUM_ENVS", "value": str(num_envs)},
                {"name": "NUM_TIMESTEPS", "value": str(num_timesteps)},
                {"name": "SEED", "value": str(seed)},
            ]
        },
    }


def launch(
    env_name: str,
    num_envs: int,
    num_timesteps: int,
    seed: int,
    job_queue: str,
    job_definition: str,
    dry_run: bool,
):
    job_request = build_job_request(
        env_name, num_envs, num_timesteps, seed, job_queue, job_definition
    )
    job_name = job_request["jobName"]

    print(f"{'='*60}")
    print(f"  AWS Batch MuJoCo RL Training (Brax PPO)")
    print(f"  Job:        {job_name}")
    print(f"  Env:        {env_name}")
    print(f"  Envs/steps: {num_envs} envs, {num_timesteps} timesteps (seed {seed})")
    print(f"  Queue:      {job_queue}")
    print(f"  Job def:    {job_definition}")
    print(f"{'='*60}")

    if dry_run:
        print("[dry-run] Would call batch.submit_job with:\n")
        print(json.dumps(job_request, indent=2))
        print("\n[dry-run] No AWS calls made.")
        return job_name

    # Imported here so --dry-run works without boto3 installed
    import boto3

    batch_client = boto3.client("batch", region_name=REGION)
    response = batch_client.submit_job(**job_request)
    job_id = response["jobId"]

    print(f"\n  Launched! Job ID: {job_id}")
    print(f"\n  Monitor:")
    print(f"    aws batch describe-jobs --jobs {job_id} --region {REGION}")
    print(f"\n  Checkpoints and rollout.mp4 upload to:")
    print(f"    s3://<checkpoints-bucket>/checkpoints/{job_id}/")
    print(
        f"  (bucket name: aws ssm get-parameter --name /{PROJECT_NAME}/checkpoints-bucket "
        f"--query Parameter.Value --output text --region {REGION})"
    )
    return job_id


def main():
    p = argparse.ArgumentParser(description="AWS Batch MuJoCo RL launcher")
    p.add_argument(
        "--env-name",
        default="CartpoleBalance",
        help="MuJoCo Playground env name (dm_control suite, locomotion or manipulation)",
    )
    p.add_argument("--num-envs", type=int, default=2048)
    p.add_argument("--num-timesteps", type=int, default=5_000_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--job-queue",
        default=None,
        help="Batch job queue name (default: from env or physical-ai-dev-mujoco-queue)",
    )
    p.add_argument(
        "--job-definition",
        default=None,
        help="Batch job definition name (default: from env or physical-ai-dev-mujoco-train)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the job; make no AWS calls",
    )
    args = p.parse_args()

    job_queue = args.job_queue or _default_queue()
    job_definition = args.job_definition or _default_job_def()

    launch(
        args.env_name,
        args.num_envs,
        args.num_timesteps,
        args.seed,
        job_queue,
        job_definition,
        args.dry_run,
    )


if __name__ == "__main__":
    main()
