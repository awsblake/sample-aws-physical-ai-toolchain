terraform {
  required_version = ">= 1.5"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region

  # Project lets the restricted deploy role mutate these resources; auto-delete
  # keeps Amazon SpringClean from deleting them after 7 days.
  default_tags {
    tags = {
      Project       = var.project_name
      Environment   = var.environment
      "auto-delete" = "no"
    }
  }
}
