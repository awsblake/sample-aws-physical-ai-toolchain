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

  # Opt out of the account's untagged-resource cleanup, which deletes
  # resources older than 7 days that lack this tag.
  default_tags {
    tags = {
      "auto-delete" = "no"
    }
  }
}
