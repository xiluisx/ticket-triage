variable "aws_account_id" {
  type        = string
  description = "Personal AWS account; the provider refuses to run against any other account"
}

variable "alert_email" {
  type        = string
  description = "Email for budget and alarm notifications"
}

variable "budget_limit_usd" {
  type    = string
  default = "5"
}

variable "github_repo" {
  type    = string
  default = "xiluisx/ticket-triage"
}
