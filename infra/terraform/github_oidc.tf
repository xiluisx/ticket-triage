# The account already has a GitHub OIDC provider (shared with other projects): reference it, never own it,
# so `terraform destroy` here cannot break other repos.
data "aws_iam_openid_connect_provider" "github" {
  url = "https://token.actions.githubusercontent.com"
}

data "aws_iam_policy_document" "gh_ci_trust" {
  statement {
    actions = ["sts:AssumeRoleWithWebIdentity"]
    principals {
      type        = "Federated"
      identifiers = [data.aws_iam_openid_connect_provider.github.arn]
    }
    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }
    condition {
      test     = "StringLike"
      variable = "token.actions.githubusercontent.com:sub"
      values   = ["repo:${var.github_repo}:*"]
    }
  }
}

data "aws_iam_policy_document" "gh_cd_trust" {
  statement {
    actions = ["sts:AssumeRoleWithWebIdentity"]
    principals {
      type        = "Federated"
      identifiers = [data.aws_iam_openid_connect_provider.github.arn]
    }
    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }
    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:sub"
      values   = ["repo:${var.github_repo}:ref:refs/heads/main"]
    }
  }
}

resource "aws_iam_role" "gh_ci" {
  name               = "ticket-triage-gh-ci"
  assume_role_policy = data.aws_iam_policy_document.gh_ci_trust.json
}

resource "aws_iam_role" "gh_cd" {
  name               = "ticket-triage-gh-cd"
  assume_role_policy = data.aws_iam_policy_document.gh_cd_trust.json
}

data "aws_iam_policy_document" "gh_ci" {
  statement {
    actions   = ["s3:GetObject"]
    resources = ["${aws_s3_bucket.data.arn}/models/*", "${aws_s3_bucket.data.arn}/datasets/*"]
  }
  statement {
    actions   = ["s3:ListBucket"]
    resources = [aws_s3_bucket.data.arn]
    condition {
      test     = "StringLike"
      variable = "s3:prefix"
      values   = ["models/*", "datasets/*"]
    }
  }
}

data "aws_iam_policy_document" "gh_cd" {
  statement {
    actions   = ["s3:ListBucket"]
    resources = ["arn:aws:s3:::${local.state_bucket}"]
  }
  statement {
    actions = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"]
    resources = [
      "arn:aws:s3:::${local.state_bucket}/main/terraform.tfstate",
      "arn:aws:s3:::${local.state_bucket}/main/terraform.tfstate.tflock",
    ]
  }
}

resource "aws_iam_role_policy" "gh_ci" {
  name   = "read-models-datasets"
  role   = aws_iam_role.gh_ci.id
  policy = data.aws_iam_policy_document.gh_ci.json
}

resource "aws_iam_role_policy" "gh_cd" {
  name   = "terraform-state"
  role   = aws_iam_role.gh_cd.id
  policy = data.aws_iam_policy_document.gh_cd.json
}
