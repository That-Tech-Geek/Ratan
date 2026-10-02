import postgres from "postgres";

if (!process.env.DATABASE_URL) throw new Error("DATABASE_URL is required");
const sql = postgres(process.env.DATABASE_URL, { max: 1, prepare: false });

const requiredTables = [
  "schema_migrations","schools","students","consents","diagnostic_sessions",
  "diagnostic_responses","likert_sessions","likert_responses","sync_events",
  "audit_logs","teachers"
];

const requiredColumns = {
  diagnostic_sessions: ["client_session_id","session_token_hash","issued_at","expires_at"],
  diagnostic_responses: ["sync_event_id"],\n  diagnostic_sessions: ["class_no","subject","question_ids"],
  consents: ["withdrawn_at"],
  audit_logs: ["school_id"],
};

try {
  const tables = new Set((await sql`
    SELECT table_name FROM information_schema.tables
    WHERE table_schema = 'public'
  `).map((r) => r.table_name));

  for (const table of requiredTables) {
    if (!tables.has(table)) throw new Error(`Missing table: ${table}`);
  }

  for (const [table, columns] of Object.entries(requiredColumns)) {
    const rows = await sql`
      SELECT column_name FROM information_schema.columns
      WHERE table_schema = 'public' AND table_name = ${table}
    `;
    const actual = new Set(rows.map((r) => r.column_name));
    for (const column of columns) {
      if (!actual.has(column)) throw new Error(`Missing column: ${table}.${column}`);
    }
  }

  const versions = await sql`SELECT version FROM schema_migrations ORDER BY version`;
  if (versions.length !== 5 || versions.at(-1)?.version !== "005_diagnostic_selection") {
    throw new Error("Unexpected migration state");
  }

  console.log("database schema verification: PASS");
} finally {
  await sql.end({ timeout: 1 });
}
