import { readdir, readFile } from "node:fs/promises";
import { join } from "node:path";
import postgres from "postgres";

const dir = "migrations";
const files = (await readdir(dir)).filter((f) => /^\d+_.+\.sql$/.test(f)).sort();
const databaseUrl =
  process.env.DATABASE_URL ??
  process.env.POSTGRES_URL_NON_POOLING ??
  process.env.POSTGRES_URL ??
  process.env.POSTGRES_PRISMA_URL;
if (!databaseUrl) {
  throw new Error(
    "A PostgreSQL connection is required: set DATABASE_URL or a Vercel/Supabase POSTGRES_* variable.",
  );
}

const sql = postgres(databaseUrl, { max: 1, prepare: false, ssl: "require" });
try {
  await sql`CREATE TABLE IF NOT EXISTS schema_migrations (
    version VARCHAR(64) PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
  )`;

  const applied = new Set((await sql`SELECT version FROM schema_migrations`).map((r) => String(r.version)));
  for (const file of files) {
    const version = file.replace(/\.sql$/, "");
    if (applied.has(version)) continue;
    const migration = await readFile(join(dir, file), "utf8");
    console.log(`Applying ${version}...`);
    await sql.begin(async (tx) => {
      await tx.unsafe(migration);
      await tx`INSERT INTO schema_migrations (version) VALUES (${version})`;
    });
  }
} finally {
  await sql.end({ timeout: 1 });
}
