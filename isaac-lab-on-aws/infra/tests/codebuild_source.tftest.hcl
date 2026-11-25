# Regression test for sampleawsphysicalaitoolchain-nln: the Isaac Lab CodeBuild
# project must get the repo checkout, since its buildspec runs
# `docker build -f containers/isaac-lab/Dockerfile .` and the Dockerfile COPYs
# containers/isaac-lab/, training/ and workflows/ from the repo root.
# Runs offline (mocked AWS) from isaac-lab-on-aws/infra:
#   terraform init -backend=false && terraform test

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

run "codebuild_source_holds_isaac_lab_build_context" {
  command = plan

  # Differs from the bucket's environment in the SSM value, so an ARN rebuilt
  # from var.environment fails the IAM assert below.
  variables {
    environment = "staging"
  }

  assert {
    condition     = aws_codebuild_project.isaac_lab.source[0].type == "S3"
    error_message = "Isaac Lab CodeBuild source must be the S3 repo archive, not NO_SOURCE (an empty build dir)"
  }

  assert {
    condition     = aws_codebuild_project.isaac_lab.source[0].location == "physical-ai-dev-datasets-000000000000/codebuild-source/isaac-lab-source.zip"
    error_message = "Isaac Lab CodeBuild source must point at the repo archive in the datasets bucket"
  }

  assert {
    condition = jsondecode(aws_iam_role_policy.codebuild.policy).Statement[0].Resource == [
      "arn:aws:s3:::physical-ai-dev-datasets-000000000000",
      "arn:aws:s3:::physical-ai-dev-datasets-000000000000/*",
    ]
    error_message = "The CodeBuild role must read the bucket the archive is uploaded to (the datasets-bucket SSM value), not one rebuilt from var.environment"
  }

  assert {
    condition     = aws_s3_object.isaac_lab_source.source == data.archive_file.isaac_lab_source.output_path && data.archive_file.isaac_lab_source.output_size > 0
    error_message = "The uploaded object must be the non-empty repo archive"
  }

  assert {
    condition     = contains([for v in aws_codebuild_project.isaac_lab.environment[0].environment_variable : "${v.name}=${v.value}"], "NGC_SECRET_NAME=physical-ai/ngc-api-key")
    error_message = "The buildspec reads the NGC key from $NGC_SECRET_NAME, so the project must set it (default physical-ai/ngc-api-key)"
  }

  assert {
    condition     = alltrue([for e in ["**/.env", "**/*.tfstate", "**/.terraform", ".git", "**/.ipynb_checkpoints", "PLAN.md", "docs/ROADMAP.md"] : contains(data.archive_file.isaac_lab_source.excludes, e)])
    error_message = "The source archive must exclude secrets and Terraform state at any depth"
  }

  assert {
    condition = alltrue([
      for f in ["containers/isaac-lab/Dockerfile", "containers/isaac-lab/sm-train-entrypoint.sh", "containers/isaac-lab/batch-train-entrypoint.sh", "containers/isaac-lab/train_entrypoint.py"] :
      fileexists("${data.archive_file.isaac_lab_source.source_dir}/${f}")
    ])
    error_message = "The source archive must be rooted at the repo so it holds containers/isaac-lab/"
  }

  assert {
    condition = alltrue([
      for d in ["training/envs", "workflows"] :
      length(fileset("${data.archive_file.isaac_lab_source.source_dir}/${d}", "**")) > 0
    ])
    error_message = "The source archive must be rooted at the repo so it holds training/ and workflows/, which the Dockerfile COPYs"
  }

  assert {
    condition = length([
      for e in data.archive_file.isaac_lab_source.excludes : e
      if startswith(e, "containers") || startswith(e, "workflows") || (startswith(e, "training") && e != "training/data/**")
    ]) == 0
    error_message = "The source archive must not exclude the Docker build context (containers/, training/, workflows/)"
  }
}
