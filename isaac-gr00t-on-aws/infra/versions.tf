terraform {
  required_version = ">= 1.5"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.0"
    }
    archive = {
      source  = "hashicorp/archive"
      version = ">= 2.5"
    }
  }
}

provider "aws" {
  region = var.aws_region

  # Amazon SpringClean deletes untagged resources older than 7 days.
  default_tags {
    tags = {
      "auto-delete" = "no"
    }
  }
}
