# Inference API: one container Lambda behind a Function URL, tickets in DynamoDB.
# Scope cut for the 2026-11-24 delivery: no API Gateway, Cognito, custom domain or SES ingest.

variable "api_image_tag" {
  type        = string
  description = "ECR image tag to deploy (git sha); empty creates only the repository"
  default     = ""
}

variable "api_key_sha256" {
  type        = string
  description = "hex SHA-256 of the demo tenant API key (the key itself never reaches Terraform)"
  default     = ""
}

locals {
  api_name        = "ticket-triage-api"
  jev_param_name  = "/ticket-triage/typesafe-api-key"
  claude_model_id = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
}

resource "aws_ecr_repository" "api" {
  name                 = local.api_name
  image_tag_mutability = "IMMUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_lifecycle_policy" "api" {
  repository = aws_ecr_repository.api.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep the last 3 images"
      selection    = { tagStatus = "any", countType = "imageCountMoreThan", countNumber = 3 }
      action       = { type = "expire" }
    }]
  })
}

resource "aws_dynamodb_table" "tickets" {
  name         = "ticket-triage-tickets"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "tenant_id"
  range_key    = "ticket_id"

  attribute {
    name = "tenant_id"
    type = "S"
  }

  attribute {
    name = "ticket_id"
    type = "S"
  }

  server_side_encryption {
    enabled = true
  }

  point_in_time_recovery {
    enabled = true
  }
}

resource "aws_cloudwatch_log_group" "api" {
  name              = "/aws/lambda/${local.api_name}"
  retention_in_days = 14
}

data "aws_iam_policy_document" "lambda_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "api" {
  name               = local.api_name
  assume_role_policy = data.aws_iam_policy_document.lambda_assume.json
}

data "aws_iam_policy_document" "api" {
  statement {
    sid       = "Logs"
    actions   = ["logs:CreateLogStream", "logs:PutLogEvents"]
    resources = ["${aws_cloudwatch_log_group.api.arn}:*"]
  }

  statement {
    sid       = "Tickets"
    actions   = ["dynamodb:PutItem", "dynamodb:GetItem", "dynamodb:Query"]
    resources = [aws_dynamodb_table.tickets.arn]
  }

  statement {
    sid       = "JevKey"
    actions   = ["ssm:GetParameter"]
    resources = ["arn:aws:ssm:us-east-1:${local.account_id}:parameter${local.jev_param_name}"]
  }

  statement {
    sid       = "JevKeyDecrypt"
    actions   = ["kms:Decrypt"]
    resources = ["arn:aws:kms:us-east-1:${local.account_id}:alias/aws/ssm"]
    condition {
      test     = "StringEquals"
      variable = "kms:ViaService"
      values   = ["ssm.us-east-1.amazonaws.com"]
    }
  }

  statement {
    sid     = "Claude"
    actions = ["bedrock:InvokeModel"]
    # The US cross-region inference profile routes to the model in us-east-1/us-east-2/us-west-2.
    resources = [
      "arn:aws:bedrock:us-east-1:${local.account_id}:inference-profile/${local.claude_model_id}",
      "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-haiku-4-5-20251001-v1:0",
      "arn:aws:bedrock:us-east-2::foundation-model/anthropic.claude-haiku-4-5-20251001-v1:0",
      "arn:aws:bedrock:us-west-2::foundation-model/anthropic.claude-haiku-4-5-20251001-v1:0",
    ]
  }
}

resource "aws_iam_role_policy" "api" {
  name   = local.api_name
  role   = aws_iam_role.api.id
  policy = data.aws_iam_policy_document.api.json
}

resource "aws_lambda_function" "api" {
  count         = var.api_image_tag == "" ? 0 : 1
  function_name = local.api_name
  role          = aws_iam_role.api.arn
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.api.repository_url}:${var.api_image_tag}"
  architectures = ["x86_64"]
  memory_size   = 2048 # more memory = more CPU: cold start (SSM + models + clients) ~11 s at 1 GB
  timeout       = 60   # Function URL has no 30 s gateway cap

  environment {
    variables = {
      TRIAGE_ENV        = "prod"
      TENANT_ID         = "cfpb-demo"
      TICKETS_TABLE     = aws_dynamodb_table.tickets.name
      API_KEY_SHA256    = var.api_key_sha256
      JEV_API_KEY_PARAM = local.jev_param_name
      CLAUDE_MODEL_ID   = local.claude_model_id
      JEV_USD_PER_CALL  = "0.0001"
    }
  }

  tracing_config {
    mode = "PassThrough"
  }

  depends_on = [aws_cloudwatch_log_group.api, aws_iam_role_policy.api]
}

# Public URL; every route except /health checks the tenant API key inside the Lambda.
resource "aws_lambda_function_url" "api" {
  count              = var.api_image_tag == "" ? 0 : 1
  function_name      = aws_lambda_function.api[0].function_name
  authorization_type = "NONE"
}

resource "aws_cloudwatch_metric_alarm" "api_errors" {
  alarm_name          = "ticket-triage-api-errors"
  alarm_description   = "Unhandled errors in the inference API"
  namespace           = "TicketTriage"
  metric_name         = "Errors"
  statistic           = "Sum"
  period              = 300
  evaluation_periods  = 1
  threshold           = 0
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alerts.arn]
}

resource "aws_cloudwatch_metric_alarm" "api_unclassified" {
  alarm_name          = "ticket-triage-unclassified"
  alarm_description   = "Tickets left unclassified (every cascade level failed)"
  namespace           = "TicketTriage"
  metric_name         = "Unclassified"
  statistic           = "Sum"
  period              = 300
  evaluation_periods  = 1
  threshold           = 0
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alerts.arn]
}

output "api_ecr_url" {
  value = aws_ecr_repository.api.repository_url
}

output "api_url" {
  value = var.api_image_tag == "" ? null : aws_lambda_function_url.api[0].function_url
}
