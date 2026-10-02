import postgres from "postgres";

if (!process.env.DATABASE_URL) throw new Error("DATABASE_URL is required");
const sql = postgres(process.env.DATABASE_URL, { max: 1, prepare: false });

try {
  const rows = await sql\`
    SELECT se.id, se.payload
    FROM sync_events se
    LEFT JOIN diagnostic_responses dr ON dr.sync_event_id = se.id
    WHERE se.entity_type = 'diagnostic_response'
      AND dr.id IS NULL
    ORDER BY se.received_at
    LIMIT 1000
  \`;

  let repaired = 0;
  await sql.begin(async (tx) => {
    for (const row of rows) {
      const payload = row.payload as Record<string, unknown>;
      const sessionId = Number(payload.session_id);
      const questionId = typeof payload.question_id === "string" ? payload.question_id : "";
      const selectedOption = typeof payload.selected_option === "string" ? payload.selected_option : "";
      const responseTimeMs = Number(payload.response_time_ms);
      if (!Number.isInteger(sessionId) || !questionId || !selectedOption || !Number.isInteger(responseTimeMs)) continue;

      await tx\`
        INSERT INTO diagnostic_responses
          (session_id, question_id, selected_option, response_time_ms, synced_at, sync_event_id)
        VALUES
          (${sessionId}, ${questionId}, ${selectedOption}, ${responseTimeMs}, NOW(), ${row.id})
        ON CONFLICT (session_id, question_id) DO NOTHING
      \`;
      repaired++;
    }
  });

  const [{syncCount}] = await sql\`
    SELECT COUNT(*)::int AS "syncCount"
    FROM sync_events
    WHERE entity_type = 'diagnostic_response'
  \`;
  const [{responseCount}] = await sql\`
    SELECT COUNT(*)::int AS "responseCount"
    FROM diagnostic_responses
  \`;
  console.log(JSON.stringify({scanned:rows.length,repaired,syncCount,responseCount}));
} finally {
  await sql.end({timeout:1});
}
