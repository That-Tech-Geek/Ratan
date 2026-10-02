# Gyaan Saathi

Gyaan Saathi is an offline-first learning diagnostics web application designed for low-end Android devices and intermittent 2G/3G connectivity.

## Architecture

```
Next.js App Router on Vercel
        |
        +-- React learning experience
        +-- Route Handlers (/api/v1/*)
        +-- Service Worker + IndexedDB
        |
        v
PostgreSQL-compatible database via DATABASE_URL
```

The repository is intentionally a single deployable application. There is no separate frontend runtime or Django service.

## Current flow

1. The student opens the Next.js web app.
2. Diagnostic questions are served from the same deployment.
3. Responses are written to IndexedDB immediately.
4. The service worker keeps the shell usable across connectivity loss.
5. The sync API accepts queued writes in batches of up to 20.
6. Server-side persistence uses PostgreSQL-compatible SQL and audit records.

## Vercel deployment

Import this repository into Vercel and use the default Next.js build settings.

Database configuration is compatible with the Vercel Supabase integration. The application accepts the following variables, in order:

- `DATABASE_URL` for an explicit application connection.
- `POSTGRES_URL` for the Vercel/Supabase pooled connection.
- `POSTGRES_PRISMA_URL` as a fallback.
- `POSTGRES_URL_NON_POOLING` as a final fallback.

The Vercel Supabase integration provisions `POSTGRES_URL`, `POSTGRES_PRISMA_URL`, and `POSTGRES_URL_NON_POOLING` automatically. The runtime uses a single Postgres.js connection per warm serverless instance, disables prepared statements, and requires TLS.

Migrations use `DATABASE_URL` first and otherwise prefer `POSTGRES_URL_NON_POOLING`, so the same repository works in GitHub Actions and against the linked Supabase project. Run `npm run migrate` once against the target Supabase database before using authenticated persistence routes.

## Development

```bash
npm install
npm run dev
npm run typecheck
npm run build
npm test
```

## MVP boundaries

Supabase is the only backend platform: Supabase Auth handles teacher OTP authentication and Supabase Postgres handles persistence. WhatsApp Business, object storage, PDF generation, and advanced reporting remain separate integrations.
