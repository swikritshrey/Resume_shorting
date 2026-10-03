# Putting resume_ online

Run it as ONE process. Runs are held in memory, so do not start several workers or copies behind a load balancer.

## Required settings (environment variables)
- `APP_PASSWORD`  : a long password. The app refuses to serve a non-local address without it. Login name is `admin` (change with `APP_USER`).
- `ALLOWED_HOSTS` : your domain, for example `resumes.example.com` (several can be separated by commas).
- `PORT`          : most hosts set this for you. When it is set the app listens on all addresses.

## Optional settings
- `DATA_DIR` : folder for saved runs (use a persistent disk, otherwise saved runs vanish on restart).
- `RETENTION_DAYS` : saved runs older than this are deleted (default 30, 0 = keep forever).
- `MAX_UPLOAD_MB` (default 1024), `MAX_PARALLEL_RUNS` (default 2).
- `ANTHROPIC_API_KEY` : turns on written assessments.

## Start
`pip install -r requirements.txt` then `python app.py` (or the `Procfile` on hosts that read it).
Health check address: `/healthz`.

## Before real students' data goes in
Use HTTPS (your host usually provides it), tell candidates how their data is used, and check the DPDP Act (2023) rules.
