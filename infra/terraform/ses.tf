# Inbound email domain for tickets. Only the tickets. subdomain is touched: the root domain keeps its
# own MX records (Cloudflare Email Routing).
resource "aws_sesv2_email_identity" "inbound" {
  email_identity = var.inbound_domain

  dkim_signing_attributes {
    next_signing_key_length = "RSA_2048_BIT"
  }
}

# Easy DKIM always returns 3 tokens; count must be static because tokens are unknown at plan time.
resource "cloudflare_dns_record" "dkim" {
  count   = 3
  zone_id = var.cloudflare_zone_id
  name    = "${aws_sesv2_email_identity.inbound.dkim_signing_attributes[0].tokens[count.index]}._domainkey.${var.inbound_domain}"
  type    = "CNAME"
  content = "${aws_sesv2_email_identity.inbound.dkim_signing_attributes[0].tokens[count.index]}.dkim.amazonses.com"
  ttl     = 1
  proxied = false
}

resource "cloudflare_dns_record" "inbound_mx" {
  zone_id  = var.cloudflare_zone_id
  name     = var.inbound_domain
  type     = "MX"
  content  = "inbound-smtp.us-east-1.amazonaws.com"
  priority = 10
  ttl      = 1
}
