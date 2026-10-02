import postgres from "postgres";

let client: ReturnType<typeof postgres> | null = null;

function databaseUrl() {
  return (
    process.env.DATABASE_URL ??
    process.env.POSTGRES_URL ??
    process.env.POSTGRES_PRISMA_URL ??
    process.env.POSTGRES_URL_NON_POOLING
  );
}

export function db() {
  const url = databaseUrl();
  if (!url) {
    throw new Error(
      "A PostgreSQL connection is required: set DATABASE_URL or a Vercel/Supabase POSTGRES_* variable.",
    );
  }
  return (
    client ??
   = postgres(url, {
      max: 1,
      prepare: false,
      ssl: "require",
    })
  );
}

export async function closeDb() {
  if (client) {
    await client.end({ timeout: 1 });
    client = null;
  }
}

export async function assertSchemaVersion(expected: string) {
  const rows = await db()`SELECT version FROM schema_migrations ORDER BY version DESC LIMIT 1`;
  if (rows[0]?.version !== expected) {
    throw new Error(`Schema mismatch: expected ${expected}, got ${rows[0]?.version ?? "none"}`);
  }
}
