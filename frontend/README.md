# Attune frontend

A deliberately minimal Vercel-ready frontend for the Attune MVP.

## Deploy

Import the repository into Vercel. No build command or environment variables are required.

Vercel serves `frontend/index.html` through `vercel.json`.

## Local

From the repository root:

    python -m http.server 8000 --directory frontend

The frontend is intentionally dependency-free. The Rust runtime and research stack remain separate from this presentation layer and can be connected through an API later.
