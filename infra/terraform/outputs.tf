output "data_bucket" {
  value = aws_s3_bucket.data.bucket
}

output "gh_ci_role_arn" {
  value = aws_iam_role.gh_ci.arn
}

output "gh_cd_role_arn" {
  value = aws_iam_role.gh_cd.arn
}

output "alerts_topic_arn" {
  value = aws_sns_topic.alerts.arn
}
