variable "aws_region" {
  description = "AWS region to deploy into"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "project_name" {
  description = "Project name prefix (must match foundation)"
  type        = string
  default     = "physical-ai"
}

variable "permissions_boundary_arn" {
  description = "ARN of the IAM policy set as permissions boundary on every role this stack creates (null for none)"
  type        = string
  default     = null
}
