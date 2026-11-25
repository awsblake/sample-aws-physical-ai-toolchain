# Regression test for sampleawsphysicalaitoolchain-nln: both GR00T CodeBuild
# projects must get the repo checkout, since their buildspecs run
# `docker build ... containers/gr00t-training/` and `cd containers/gr00t-inference`.
# Runs offline (mocked AWS) from isaac-gr00t-on-aws/infra:
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

run "codebuild_source_holds_gr00t_build_context" {
  command = plan

  assert {
    condition     = aws_codebuild_project.groot_training.source[0].type == "S3" && aws_codebuild_project.groot_inference.source[0].type == "S3"
    error_message = "GR00T CodeBuild sources must be the S3 repo archive, not NO_SOURCE (an empty build dir)"
  }

  assert {
    condition = alltrue([
      for p in [aws_codebuild_project.groot_training, aws_codebuild_project.groot_inference] :
      p.source[0].location == "physical-ai-dev-datasets-000000000000/codebuild-source/gr00t-source.zip"
    ])
    error_message = "GR00T CodeBuild sources must point at the repo archive in the datasets bucket"
  }

  assert {
    condition     = aws_s3_object.gr00t_source.source == data.archive_file.gr00t_source.output_path && data.archive_file.gr00t_source.output_size > 0
    error_message = "The uploaded object must be the non-empty repo archive"
  }

  assert {
    condition     = alltrue([for e in ["**/.env", "**/*.tfstate", "**/.terraform", ".git", "**/.ipynb_checkpoints", "PLAN.md", "docs/ROADMAP.md"] : contains(data.archive_file.gr00t_source.excludes, e)])
    error_message = "The source archive must exclude secrets and Terraform state at any depth"
  }

  assert {
    condition = alltrue([
      for f in ["containers/gr00t-training/Dockerfile", "containers/gr00t-training/train_entrypoint.py", "containers/gr00t-training/batch-train-entrypoint.sh", "containers/gr00t-inference/Dockerfile", "containers/gr00t-inference/serve.py"] :
      fileexists("${data.archive_file.gr00t_source.source_dir}/${f}")
    ])
    error_message = "The source archive must be rooted at the repo so it holds containers/gr00t-training/ and containers/gr00t-inference/"
  }

  assert {
    condition     = length([for e in data.archive_file.gr00t_source.excludes : e if startswith(e, "containers")]) == 0
    error_message = "The source archive must not exclude the Docker build contexts under containers/"
  }
}
