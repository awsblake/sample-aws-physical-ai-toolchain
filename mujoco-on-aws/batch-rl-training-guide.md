# MuJoCo — Batch RL Training Guide

Train a policy with Brax PPO on a [MuJoCo Playground](https://playground.mujoco.org/) env as an AWS Batch job. Each job runs on one GPU, writes checkpoints and a rollout MP4 to the foundation checkpoints bucket, and the compute environment scales back to zero when the queue is empty.

> **Preview.** These steps have not yet been run end to end on hardware. Throughput, wall time and cost figures will be recorded here after the first validation run.

---

## Overview

```
launch_mujoco_batch.py ──submit_job──▶ Batch queue ──▶ compute env (g6e.4xlarge, 0 → 1 instance)
                                                              │
                                                              ▼
                                         physical-ai/mujoco:latest  batch-train
                                         (Brax PPO on MJX, MUJOCO_GL=egl)
                                                              │
                                                              ▼
                                  s3://<checkpoints-bucket>/checkpoints/<job_id>/
                                      checkpoints/<step>/   rollout.mp4
```

| Piece | Where |
|-------|-------|
| Compute environment, queue, job definition | [`infra/batch.tf`](infra/batch.tf) |
| Training entrypoint | [`containers/mujoco/train_entrypoint.py`](../containers/mujoco/train_entrypoint.py) |
| Launcher | [`training/scripts/launch_mujoco_batch.py`](../training/scripts/launch_mujoco_batch.py) |

Unlike the [Isaac Lab Batch guide](../isaac-lab-on-aws/batch-rl-training-guide.md), there is no NGC key to store and no job definition to register by hand: Terraform creates the job definition.

---

## Prerequisites

- AWS CLI configured with credentials for the target account
- Terraform >= 1.5
- Python 3 with `boto3` (for the launcher)
- EC2 quota: **Running On-Demand G and VT instances** of at least 16 vCPUs in the deploy region (one g6e.4xlarge)

All commands below run from the repo root and use `us-east-2`; use the region foundation is deployed in.

---

## Step 1: Deploy Foundation Infrastructure

Foundation creates the VPC, the checkpoints and models buckets, and the `physical-ai/mujoco` ECR repository. Skip this step if it is already deployed with Terraform.

```bash
terraform -chdir=foundation/infra init
terraform -chdir=foundation/infra apply -var="aws_region=us-east-2"
```

See the [foundation README](../foundation/README.md) for what it creates.

---

## Step 2: Deploy MuJoCo Batch Infrastructure

```bash
terraform -chdir=mujoco-on-aws/infra init
terraform -chdir=mujoco-on-aws/infra apply -var="aws_region=us-east-2" -var="enable_batch=true"
```

Add `-var="enable_sim=true"` if the [headless instance](headless-sim-guide.md) is deployed from the same root, or Terraform removes it.

**What this creates:**

| Resource | Purpose |
|----------|---------|
| CodeBuild: `physical-ai-dev-mujoco-build` | Builds the MuJoCo image (no NGC key) |
| Batch compute env: `physical-ai-dev-mujoco-compute` | g6e.4xlarge in the foundation private subnets, `min_vcpus = 0` |
| Batch job queue: `physical-ai-dev-mujoco-queue` | Job scheduling |
| Batch job definition: `physical-ai-dev-mujoco-train` | Single node, 1 GPU, runs `batch-train`, sets `CHECKPOINT_BUCKET` |
| Security group | Egress only (single-node jobs need no inbound traffic) |
| IAM roles | Instance role (ECS agent, image pull, logs); job role with S3 read/write on the checkpoints and models buckets only |

**Verify:**

```bash
terraform -chdir=mujoco-on-aws/infra output
# batch_job_definition = "physical-ai-dev-mujoco-train"
# batch_job_queue = "physical-ai-dev-mujoco-queue"
# codebuild_project = "physical-ai-dev-mujoco-build"
```

### Configuration Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `enable_batch` | `false` | Set to `true` to deploy the Batch resources |
| `batch_instance_type` | `g6e.4xlarge` | GPU instance type (1 L40S GPU) |
| `batch_max_vcpus` | `32` | Maximum vCPUs the compute environment scales to |
| `batch_job_vcpus` | `8` | vCPUs reserved per job |
| `batch_job_memory_mib` | `32768` | Memory reserved per job |
| `batch_job_timeout_seconds` | `14400` | Per-attempt timeout, so a hung job does not keep an instance billing |

---

## Step 3: Build the Training Container

The image builds in CodeBuild; no local Docker is needed. This guide needs the Terraform foundation from Step 1: the CDK Foundation stack publishes none of the `/physical-ai/*` SSM parameters Step 2 reads.

### 3a. Package and upload the source

```bash
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# From repo root
zip -r /tmp/source.zip . \
  -x ".git/*" \
  -x "*/node_modules/*" \
  -x "*/.terraform/*" \
  -x "*/terraform.tfstate*" \
  -x ".claude/*"

aws s3 cp /tmp/source.zip \
  "s3://physical-ai-dev-datasets-${ACCOUNT_ID}/codebuild-source/source.zip" \
  --region us-east-2
```

### 3b. Trigger the build

```bash
aws codebuild start-build \
  --project-name "physical-ai-dev-mujoco-build" \
  --region us-east-2 \
  --source-type-override S3 \
  --source-location-override \
    "physical-ai-dev-datasets-${ACCOUNT_ID}/codebuild-source/source.zip" \
  --query 'build.id' --output text
```

### 3c. Monitor progress

```bash
aws codebuild batch-get-builds \
  --ids "<BUILD_ID>" \
  --region us-east-2 \
  --query 'builds[0].{status:buildStatus,phase:currentPhase}'
```

**What it produces:** `<ACCOUNT_ID>.dkr.ecr.us-east-2.amazonaws.com/physical-ai/mujoco:latest`, the tag the job definition runs.

---

## Step 4: Submit a Training Job

Preview the request first. `--dry-run` prints the SubmitJob payload (shown compacted here) and makes no AWS calls:

```bash
python training/scripts/launch_mujoco_batch.py --dry-run
```

```json
{
  "jobName": "mujoco-rl-<timestamp>",
  "jobQueue": "physical-ai-dev-mujoco-queue",
  "jobDefinition": "physical-ai-dev-mujoco-train",
  "containerOverrides": {
    "environment": [
      {"name": "ENV_NAME", "value": "CartpoleBalance"},
      {"name": "NUM_ENVS", "value": "2048"},
      {"name": "NUM_TIMESTEPS", "value": "5000000"},
      {"name": "SEED", "value": "0"}
    ]
  }
}
```

Then submit it, reading the queue and job definition from the Terraform outputs:

```bash
AWS_DEFAULT_REGION=us-east-2 python training/scripts/launch_mujoco_batch.py \
  --env-name CartpoleBalance \
  --num-envs 2048 \
  --num-timesteps 5000000 \
  --job-queue "$(terraform -chdir=mujoco-on-aws/infra output -raw batch_job_queue)" \
  --job-definition "$(terraform -chdir=mujoco-on-aws/infra output -raw batch_job_definition)"
```

Without `--job-queue` and `--job-definition`, the launcher uses `MUJOCO_BATCH_QUEUE` and `MUJOCO_BATCH_JOB_DEF` from the environment, else the default names above (built from `PROJECT_NAME` and `ENVIRONMENT`, default `physical-ai` and `dev`).

### Training Parameters

| Flag | Env var in the job | Default | Description |
|------|--------------------|---------|-------------|
| `--env-name` | `ENV_NAME` | `CartpoleBalance` | Playground env: a dm_control suite, locomotion or manipulation env name |
| `--num-envs` | `NUM_ENVS` | `2048` | Parallel MJX environments on the GPU; must divide the PPO `batch_size * num_minibatches` (a power of two is safe) |
| `--num-timesteps` | `NUM_TIMESTEPS` | `5000000` | Total environment steps |
| `--seed` | `SEED` | `0` | Random seed |
| — | `CHECKPOINT_BUCKET` | foundation checkpoints bucket | Set by the job definition |
| — | `CHECKPOINT_PREFIX` | `checkpoints/<job_id>` | S3 key prefix |

PPO hyperparameters come from Playground's tuned config for the env.

### Monitor the job

```bash
aws batch describe-jobs \
  --jobs "<JOB_ID>" \
  --region us-east-2 \
  --query 'jobs[0].{status:status,reason:statusReason,log:container.logStreamName}'
```

Job lifecycle: `SUBMITTED` → `PENDING` → `RUNNABLE` → `STARTING` → `RUNNING` → `SUCCEEDED`. The first job after an idle period waits in `RUNNABLE` for a few minutes while Batch launches an instance.

Logs go to CloudWatch Logs under `/aws/batch/job`, in the stream the command above prints.

---

## Step 5: Verify Checkpoints in S3

```bash
CHECKPOINTS_BUCKET=$(aws ssm get-parameter --name /physical-ai/checkpoints-bucket \
  --query Parameter.Value --output text --region us-east-2)

aws s3 ls "s3://${CHECKPOINTS_BUCKET}/checkpoints/<JOB_ID>/" --recursive --region us-east-2
```

You should see:

- `checkpoints/<step>/` — one Brax (orbax) checkpoint per evaluation, each with a `ppo_network_config.json`; the highest step is the final policy
- `rollout.mp4` — one episode of the trained policy, rendered offscreen

The checkpoints bucket expires objects after 14 days. Copy a policy you want to keep to the models bucket (`/physical-ai/models-bucket`). To play a checkpoint back, see the [Headless Simulation Guide](headless-sim-guide.md#step-5-play-back-a-checkpoint-to-mp4).

---

## What You'll See During Training

The entrypoint prints a header, then one line per evaluation:

```
============================================================
  MuJoCo Playground RL Training (Brax PPO)
  Env:         CartpoleBalance
  Num envs:    2048
  ...
  JAX devices: [CudaDevice(id=0)]
============================================================
  step 0: eval reward ...
  step 655360: eval reward ...
Training complete. Checkpoints saved to /tmp/checkpoints/<job_id>/checkpoints
  Rollout: ... frames -> /tmp/checkpoints/<job_id>/rollout.mp4
Done. Uploaded ... files to s3://<checkpoints-bucket>/checkpoints/<job_id>/
```

- **JAX devices** should list a `CudaDevice`. A `CpuDevice` means the container does not see the GPU.
- **eval reward** should climb across evaluations.

---

## Managing Compute Costs

The compute environment has `min_vcpus = 0`: Batch launches an instance when a job is queued and terminates it once the queue is empty, so an idle deployment costs nothing. A running job bills the g6e.4xlarge on-demand rate (~$3.00/hr in us-east-2). The per-attempt timeout (`batch_job_timeout_seconds`) stops a hung job from billing indefinitely.

---

## Troubleshooting

| Problem | Cause | Solution |
|---------|-------|----------|
| Job stuck in `RUNNABLE` | No g6e capacity in the foundation private subnets' AZs, or the G and VT vCPU quota is below 16 | Check the quota in Service Quotas; try another `batch_instance_type` that the region offers |
| `CannotPullContainerError` | The image has not been built yet, or was built in another region | Run Step 3 in the region the job runs in |
| `ValueError: Unknown Playground env` | `--env-name` is not a Playground registry name | Use a name from Playground's dm_control suite, locomotion or manipulation registry, for example `CartpoleBalance` |
| `ValueError: NUM_ENVS=... must divide batch_size * num_minibatches` | The env count does not divide the PPO batch | Use a power of two for `--num-envs` |
| `WARNING: CHECKPOINT_BUCKET not set` | The job ran with a job definition that has no bucket | Re-apply `mujoco-on-aws/infra` so the job definition reads the foundation bucket |
| Job fails after `Training complete` | Rendering the rollout failed | The checkpoints are still uploaded; check the log for the EGL error |

---

## Teardown

```bash
# Destroy the MuJoCo infra (the compute environment is already at 0 instances when idle)
terraform -chdir=mujoco-on-aws/infra destroy -var="aws_region=us-east-2" -var="enable_batch=true"

# Destroy foundation (optional — other components may depend on it)
terraform -chdir=foundation/infra destroy -var="aws_region=us-east-2"
```

---

## Architecture Notes

- **Why one GPU per job?** MJX runs all envs of a job on one GPU, so there is no multi-node traffic and no NCCL setup. To use more GPUs, submit more jobs (several seeds or envs).
- **Why its own queue?** Isaac Lab's queue is optional, keeps 16 vCPUs warm and carries a 512 GiB launch template. A separate queue keeps the two components independent.
- **Why no larger root volume?** The image is a few GB, so the default ECS GPU AMI volume is enough.
