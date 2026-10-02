import postgres from "postgres";

let client: ReturnType<typeof postgres> | null = null;

export function db() {
  if (!process.env.DATABASE_URL) throw new Error("DATABASE_URL is required");
  return client ??= postgres(process.env.DATABASE_URL, { max: 5, prepare: false });
}

export async function closeDb() {
  if (client) {
    await client.end({ timeout: 1 });
    client = null;
  }
}

export async function assertSchemaVersion(expected: string) {
  const rows = await db()\`SELECT version FROM schema_migrations ORDER BY version DESC LIMIT 1\`;
  if (rows[0]?.version !== expected) {
    throw new Error(`Schema mismatch: expected ${expected}, got ${rows[0]?.version ?? "none"}`);
  }
}
