# Attune frontend

A deliberately minimal Vercel-ready frontend for the Attune MVP.

## Deploy

Import the repository into Vercel. No build command or environment variables are required.

Vercel serves `frontend/index.html` through `vercel.json`.

## Local

From the repository root:

    python -m http.server 8000 --directory frontend

The frontend is intentionally dependency-free. The Rust runtime and research stack remain separate from this presentation layer and can be connected through an API later.


## Services

The Vercel project contains two services: the public Next.js app and an internal FastAPI clinician-tools service. The app declares a `CLINICIAN_TOOLS_URL` service binding and proxies a health check through `/api/status`. The clinician service has no public rewrite, so its URL is never exposed directly.
