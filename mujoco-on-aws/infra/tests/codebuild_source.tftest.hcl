# Regression test for sampleawsphysicalaitoolchain-71k: the MuJoCo CodeBuild
# project must get the repo checkout, since its buildspec runs
# `docker build -f containers/mujoco/Dockerfile .` and ./containers/mujoco/test-image.sh.
# Runs offline (mocked AWS) from mujoco-on-aws/infra:
#   terraform init -backend=false && terraform test
# check_source_archive.py runs this, then checks the zip it built holds the
# build context and none of the excluded secrets or state.

mock_provider "aws" {
  mock_data "aws_caller_identity" {
    defaults = {
      account_id = "000000000000"
    }
  }

  mock_data "aws_ssm_parameter" {
    defaults = {
      value = "physical-ai-dev-datasets-000000000000"
    }
  }
}

override_data {
  target = data.aws_ssm_parameter.mujoco_ecr
  values = {
    value = "000000000000.dkr.ecr.us-east-1.amazonaws.com/physical-ai/mujoco"
  }
}

run "codebuild_source_holds_mujoco_build_context" {
  command = plan

  # Differs from the bucket's environment in the SSM value, so an ARN rebuilt
  # from var.environment fails the IAM assert below.
  variables {
    environment = "staging"
  }

  assert {
    condition     = aws_codebuild_project.mujoco.source[0].type == "S3"
    error_message = "MuJoCo CodeBuild source must be the S3 repo archive, not NO_SOURCE (an empty build dir)"
  }

  assert {
    condition     = aws_codebuild_project.mujoco.source[0].location == "physical-ai-dev-datasets-000000000000/codebuild-source/mujoco-source.zip"
    error_message = "MuJoCo CodeBuild source must point at the repo archive in the datasets bucket"
  }

  assert {
    condition = jsondecode(aws_iam_role_policy.codebuild.policy).Statement[0].Resource == [
      "arn:aws:s3:::physical-ai-dev-datasets-000000000000",
      "arn:aws:s3:::physical-ai-dev-datasets-000000000000/*",
    ]
    error_message = "The CodeBuild role must read the bucket the archive is uploaded to (the datasets-bucket SSM value), not one rebuilt from var.environment"
  }

  assert {
    condition     = aws_s3_object.mujoco_source.source == data.archive_file.mujoco_source.output_path && data.archive_file.mujoco_source.output_size > 0
    error_message = "The uploaded object must be the non-empty repo archive"
  }

  assert {
    condition     = alltrue([for e in ["**/.env", "**/*.tfstate", "**/.terraform", ".git"] : contains(data.archive_file.mujoco_source.excludes, e)])
    error_message = "The source archive must exclude secrets and Terraform state at any depth"
  }

  assert {
    condition = alltrue([
      for f in ["containers/mujoco/Dockerfile", "containers/mujoco/test-image.sh", "containers/mujoco/train_entrypoint.py", "containers/mujoco/batch-train-entrypoint.sh"] :
      fileexists("${data.archive_file.mujoco_source.source_dir}/${f}")
    ])
    error_message = "The source archive must be rooted at the repo so it holds containers/mujoco/"
  }

  assert {
    condition = length([
      for e in data.archive_file.mujoco_source.excludes : e
      if startswith(e, "containers") || (startswith(e, "training") && e != "training/data/**")
    ]) == 0
    error_message = "The source archive must not exclude the Docker build context (containers/, training/)"
  }
}
