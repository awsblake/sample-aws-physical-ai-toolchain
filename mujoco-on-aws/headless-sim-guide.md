# MuJoCo — Headless Simulation Guide

Run MuJoCo simulation, training, policy playback and MP4 rendering on one GPU instance with no display. You reach the instance over SSM Session Manager; it has no inbound ports and no public IP.

> **Preview.** These steps have not yet been run end to end on hardware. Throughput figures will be recorded here after the first validation run.

---

## Overview

| | |
|-|-|
| Instance | g6e.xlarge (1 L40S GPU) in a foundation private subnet |
| AMI | AWS Deep Learning Base GPU AMI (NVIDIA driver, Docker, NVIDIA container toolkit), resolved from its public SSM parameter; no Marketplace subscription |
| Access | `aws ssm start-session` only; the security group has no inbound rules |
| Image | User data logs in to ECR and pulls `physical-ai/mujoco:latest` at first boot |
| S3 | Read-only on the foundation checkpoints and models buckets |
| Rendering | Offscreen through EGL (`MUJOCO_GL=egl`), written to MP4 |

The Terraform is [`infra/ec2.tf`](infra/ec2.tf). This mode mirrors the [Isaac Sim headless mode](../isaac-sim-on-aws/headless-mode-guide.md), minus the manual ECR pull and the DCV/SSH ports.

---

## Prerequisites

- Foundation deployed with Terraform ([foundation](../foundation/README.md)); the CDK Foundation stack publishes none of the SSM parameters this root reads
- The [Session Manager plugin](https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-working-with-install-plugin.html) for the AWS CLI on your machine
- EC2 quota: **Running On-Demand G and VT instances** of at least 4 vCPUs (one g6e.xlarge)

User data pulls the image once, at first boot, so build it before you create the instance. The CodeBuild project lives in the same Terraform root, so apply the root without `enable_sim` first, build the image ([Batch guide Step 3](batch-rl-training-guide.md#step-3-build-the-training-container)), then apply again with `enable_sim=true` (Step 1 below). If the instance came up first, pull the image by hand ([Start / Stop](#start--stop-the-instance)).

---

## Step 1: Deploy the Instance

```bash
terraform -chdir=mujoco-on-aws/infra init
terraform -chdir=mujoco-on-aws/infra apply -var="aws_region=us-east-2" -var="enable_sim=true"
```

Add `-var="enable_batch=true"` if Batch is deployed from the same root, or Terraform removes it.

| Variable | Default | Description |
|----------|---------|-------------|
| `enable_sim` | `false` | Set to `true` to create the instance |
| `sim_instance_type` | `g6e.xlarge` | GPU instance type; must be offered in the first foundation private subnet's AZ |
| `sim_volume_size` | `100` | Root volume in GB (encrypted gp3) |
| `sim_ami_ssm_parameter` | Deep Learning Base GPU AMI (Amazon Linux 2023) | Public SSM parameter for the AMI |

---

## Step 2: Connect

```bash
terraform -chdir=mujoco-on-aws/infra output -raw sim_ssm_command
# aws ssm start-session --target i-0123456789abcdef0 --region us-east-2
```

Run the printed command. Give the instance 2-3 minutes after launch for the SSM agent to register. Then, inside the session:

```bash
sudo su - ec2-user
export AWS_DEFAULT_REGION=us-east-2
IMAGE="$(aws ssm get-parameter --name /physical-ai/ecr/mujoco \
  --query Parameter.Value --output text):latest"
```

The rest of this guide runs inside that session. In a new session, or after a stop and start, run these three lines again.

---

## Step 3: Check the GPU and the Image

```bash
nvidia-smi
sudo docker images | grep physical-ai/mujoco

sudo docker run --rm --gpus all "$IMAGE" python -c \
  "import mujoco, jax; print(mujoco.__version__, jax.__version__, jax.devices())"
```

The last line should list a `CudaDevice`. If `docker images` shows nothing, the first-boot pull did not finish; see [Troubleshooting](#troubleshooting).

---

## Step 4: Train Interactively

`batch-train` is the same entrypoint the Batch job runs. Without `CHECKPOINT_BUCKET` it writes to `OUTPUT_DIR` only (the instance role cannot write to S3):

```bash
mkdir -p ~/runs/local
sudo docker run --rm --gpus all \
  -e ENV_NAME=CartpoleBalance \
  -e NUM_ENVS=2048 \
  -e NUM_TIMESTEPS=5000000 \
  -e OUTPUT_DIR=/runs/local \
  -v ~/runs:/runs \
  "$IMAGE" batch-train
```

Output lands in `~/runs/local/checkpoints/<step>/` and `~/runs/local/rollout.mp4`. The parameters are the same env vars as a Batch job ([Training Parameters](batch-rl-training-guide.md#training-parameters)).

---

## Step 5: Play Back a Checkpoint to MP4

Copy a Batch job's output from the checkpoints bucket:

```bash
CHECKPOINTS_BUCKET=$(aws ssm get-parameter --name /physical-ai/checkpoints-bucket \
  --query Parameter.Value --output text)
aws s3 cp --recursive "s3://${CHECKPOINTS_BUCKET}/checkpoints/<JOB_ID>/" ~/runs/<JOB_ID>/
```

Then load the final checkpoint and render one episode. Set `ENV_NAME` to the env the job trained on:

```bash
sudo docker run --rm -i --gpus all \
  -e ENV_NAME=CartpoleBalance \
  -e RUN_DIR=/runs/<JOB_ID> \
  -v ~/runs:/runs \
  "$IMAGE" python - <<'EOF'
import os
from pathlib import Path

import jax
import mediapy
from brax.training.agents.ppo import checkpoint
from mujoco_playground import registry

import sys
sys.path.insert(0, "/opt/ml/code")
from train_entrypoint import ppo_config

run_dir = Path(os.environ["RUN_DIR"])
# Brax writes one checkpoint per eval, named by step; the last is the final policy
ckpt = sorted(p for p in (run_dir / "checkpoints").iterdir() if p.is_dir())[-1]
env_name = os.environ["ENV_NAME"]
env = registry.load(env_name)
# Same episode length the training job's rollout used
episode_length = ppo_config(env_name).episode_length
policy = jax.jit(checkpoint.load_policy(ckpt, deterministic=True))
reset_fn, step_fn = jax.jit(env.reset), jax.jit(env.step)

rng = jax.random.PRNGKey(0)
state = reset_fn(rng)
trajectory = [state]
for _ in range(episode_length):
    rng, act_rng = jax.random.split(rng)
    action, _ = policy(state.obs, act_rng)
    state = step_fn(state, action)
    trajectory.append(state)
    if state.done:
        break

frames = env.render(trajectory, height=240, width=320)
mediapy.write_video(str(run_dir / "playback.mp4"), frames, fps=1.0 / env.dt)
print(f"{len(frames)} frames -> {run_dir / 'playback.mp4'} from {ckpt.name}")
EOF
```

This is the rollout loop from [`train_entrypoint.py`](../containers/mujoco/train_entrypoint.py) (`write_rollout`, with `ppo_config` imported from the image), with the policy loaded from disk instead of from training.

---

## Step 6: Get the MP4 Off the Instance

The instance has no inbound ports and its role cannot write to S3, so fetch files through an SSM port forward. On the instance:

```bash
cd ~/runs && python3 -m http.server 8000 --bind 127.0.0.1
```

On your machine, in a second terminal:

```bash
aws ssm start-session --target <INSTANCE_ID> --region us-east-2 \
  --document-name AWS-StartPortForwardingSession \
  --parameters 'portNumber=8000,localPortNumber=8000'
```

Then open `http://localhost:8000/` and download the MP4. Stop the server with Ctrl-C when done.

---

## Start / Stop the Instance

The instance bills per second while running (~$1.86/hr for g6e.xlarge). Stop it when you are done; the 100 GB volume (~$8/month) and the pulled image persist.

```bash
INSTANCE_ID=$(terraform -chdir=mujoco-on-aws/infra output -raw sim_instance_id)
aws ec2 stop-instances --instance-ids "$INSTANCE_ID" --region us-east-2
aws ec2 start-instances --instance-ids "$INSTANCE_ID" --region us-east-2
```

User data runs only at first boot. After a rebuild of the image, pull it again inside a session:

```bash
aws ecr get-login-password | sudo docker login --username AWS --password-stdin "${IMAGE%%/*}"
sudo docker pull "$IMAGE"
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `aws ssm start-session` returns `TargetNotConnected` | The SSM agent has not registered yet, or the Session Manager plugin is missing locally | Wait 2-3 minutes after launch; check the instance shows **Online** in Systems Manager → Fleet Manager; install the plugin |
| `terraform apply` fails with `Unsupported` for the instance type | The type is not offered in the first foundation private subnet's AZ | Set `-var="sim_instance_type=<type>"` to a GPU type that AZ offers |
| `docker images` shows no `physical-ai/mujoco` image | The image was not built before first boot, or the pull failed (user data retries 5 times) | Check `/var/log/cloud-init-output.log`, then pull by hand (see [Start / Stop](#start--stop-the-instance)) |
| `jax.devices()` lists only `CpuDevice` | The container started without the GPU | Keep `--gpus all` on every `docker run` |
| Rendering fails with an EGL error | The container cannot reach the NVIDIA EGL driver | Keep `--gpus all`; the image sets `NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics`, so do not override it |
| `aws s3 cp` returns `AccessDenied` | The role reads only the foundation checkpoints and models buckets, and cannot write | Read from those buckets; fetch outputs through [Step 6](#step-6-get-the-mp4-off-the-instance) |

---

## Teardown

```bash
terraform -chdir=mujoco-on-aws/infra destroy -var="aws_region=us-east-2" -var="enable_sim=true"
```
