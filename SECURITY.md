# Security policy

## Supported version

Security fixes target the latest minor release.

## Reporting

Open a private GitHub security advisory rather than a public issue. Do not include real meeting media, tokens, or credentials in reports.

## Deployment boundary

The v0.1 service has no authentication. It binds to `127.0.0.1` by default and must be placed behind authenticated TLS termination, request limits, and media authorization before network exposure.

## Data handling

- Keep raw media outside Git and encrypt it at rest when it contains private meetings.
- Use anonymous speaker IDs unless participants have explicitly consented to name mapping.
- Treat `source_uri` as sensitive; production implementations should return signed, short-lived replay URLs.
- Store Hugging Face and GitHub tokens only in environment variables or a secret manager.
- Parameterized SQL is mandatory; uploaded fixture size and media type limits must be enforced at the gateway.
