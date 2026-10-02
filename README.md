# Gyaan Saathi

Offline-first learning diagnostics for low-end Android devices and intermittent 2G/3G connectivity.

## Architecture

```
Student / Teacher / Volunteer PWA
        |
        v
Service Worker + IndexedDB
        |
        v
Sync queue (20 items, retry <= 5)
        |
 HTTPS when online
        v
Django REST API
        |
        v
PostgreSQL
```

The browser owns the offline experience. The server is the durable system of record. Diagnostic and Likert responses are append-oriented and the sync API is idempotency-aware.

## Implemented MVP data flow

- React + Vite PWA shell with installable service worker.
- Dexie/IndexedDB queue for offline writes.
- Queue flush on app open, `online`, and `visibilitychange`.
- Django API contracts for health, diagnostic questions, and batch sync.
- Relational models for schools, anonymised students, consent, diagnostic sessions/responses, Likert sessions/responses, learning preferences, gap reports, re-checks, guidance cards, and audit logs.
- CI runs frontend typecheck/build, backend Django tests, and cross-layer API contract checks.

## Deliberate boundaries

The source specification calls for Firebase phone OTP, WhatsApp Business API, S3-compatible storage, PDF generation, and managed PostgreSQL. Those external integrations are not faked in this migration. They should be added behind explicit service interfaces after the core offline/sync path is stable.

No student login, open-ended AI tutor, real-time chat, gamification, location tracking, or native app is part of this MVP boundary.

## Development

Frontend:
```
npm install
npm run dev
```

Backend:
```
cd backend
python -m venv .venv
pip install -r requirements.txt
python manage.py migrate
python manage.py test
```

Environment variables for production include `DJANGO_SECRET_KEY`, `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_HOST`, `POSTGRES_PORT`, and optionally `VITE_API_URL`.
