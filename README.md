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

Required environment variable:

- `DATABASE_URL`: PostgreSQL connection string from a Vercel-compatible Postgres provider.

The application lazily creates its MVP tables on the first persistence request. For production, move this schema into a managed migration pipeline once the database provider is fixed.

## Development

```bash
npm install
npm run dev
npm run typecheck
npm run build
npm test
```

## MVP boundaries

Firebase OTP, WhatsApp Business, object storage, PDF generation, production authentication/authorization, teacher dashboards, and advanced reporting are explicit next-phase integrations rather than mocked dependencies.
