import { db } from "../../../../../lib/server/db";
import { requireTeacher } from "../../../../../lib/server/auth";
import { hashSessionToken } from "../../../../../lib/server/session-token";
import { createHash } from "node:crypto";

export const runtime = "nodejs";

const MAX_ITEMS = 20;
const MAX_ITEM_BYTES = 32000;

function stableHash(value: unknown) {
  return createHash("sha256").update(JSON.stringify(value)).digest("hex");
}

function invalid(message: string) {
  return Response.json({error: message}, {status: 400});
}

export async function POST(request: Request) {
  try {
    const teacher = await requireTeacher(request);
    const raw = await request.text();
    if (raw.length > MAX_ITEMS * MAX_ITEM_BYTES) return invalid("payload_too_large");

    let body: unknown;
    try { body = JSON.parse(raw); } catch { return invalid("invalid_json"); }
    if (!body || typeof body !== "object") return invalid("invalid_body");

    const input = body as {session_id?: unknown; session_token?: unknown; items?: unknown};
    const sessionId = Number(input.session_id);
    const sessionToken = typeof input.session_token === "string" ? input.session_token : "";
    const items = input.items;

    if (!Number.isInteger(sessionId) || sessionId <= 0 || !sessionToken || !Array.isArray(items) || items.length > MAX_ITEMS) {
      return invalid("invalid_batch");
    }

    const sessionRows = await db()\`
      SELECT ds.id, ds.student_id, s.school_id, ds.expires_at, ds.session_token_hash
      FROM diagnostic_sessions ds
      JOIN students s ON s.id = ds.student_id
      JOIN teachers t ON t.school_id = s.school_id
      WHERE ds.id = ${sessionId}
        AND t.id = ${teacher.teacherId}
        AND t.active = TRUE
      LIMIT 1
    \`;
    const session = sessionRows[0];
    if (!session) return Response.json({error:"forbidden"}, {status:403});
    if (new Date(session.expires_at).getTime() <= Date.now()) return Response.json({error:"session_expired"}, {status:401});
    if (hashSessionToken(sessionToken) !== session.session_token_hash) return Response.json({error:"invalid_session_token"}, {status:401});

    const questionIds = new Set(
      ((await db()\`SELECT question_ids FROM diagnostic_sessions WHERE id = ${sessionId}\`)[0]?.question_ids ?? []) as string[],
    );

    const accepted: string[] = [];
    const duplicate: string[] = [];
    const rejected: Array<{id: string; reason: string}> = [];

    await db().begin(async (tx) => {
      for (const rawItem of items) {
        if (!rawItem || typeof rawItem !== "object") {
          rejected.push({id:"unknown",reason:"invalid_item"});
          continue;
        }

        const item = rawItem as Record<string, unknown>;
        const id = typeof item.id === "string" ? item.id : "";
        const entity = typeof item.entity === "string" ? item.entity : "";
        const action = typeof item.action === "string" ? item.action : "";
        const payload = item.payload;

        if (!/^[0-9a-f-]{36}$/i.test(id)) { rejected.push({id:id || "unknown",reason:"invalid_id"}); continue; }
        if (entity !== "diagnostic_response" || action !== "create") { rejected.push({id,reason:"unsupported_entity"}); continue; }
        if (!payload || typeof payload !== "object") { rejected.push({id,reason:"invalid_payload"}); continue; }
        if (JSON.stringify(item).length > MAX_ITEM_BYTES) { rejected.push({id,reason:"item_too_large"}); continue; }

        const response = payload as Record<string, unknown>;
        const questionId = typeof response.question_id === "string" ? response.question_id : "";
        const selectedOption = typeof response.selected_option === "string" ? response.selected_option : "";
        const responseTimeMs = Number(response.response_time_ms);

        if (!questionIds.has(questionId)) { rejected.push({id,reason:"question_not_in_session"}); continue; }
        if (!selectedOption || selectedOption.length > 128) { rejected.push({id,reason:"invalid_selected_option"}); continue; }
        if (!Number.isInteger(responseTimeMs) || responseTimeMs < 0 || responseTimeMs > 60 * 60 * 1000) { rejected.push({id,reason:"invalid_response_time"}); continue; }

        const payloadHash = stableHash(item);
        const eventRows = await tx\`
          INSERT INTO sync_events
            (idempotency_key, entity_type, action, payload_hash, payload, received_at)
          VALUES
            (${id}, 'diagnostic_response', 'create', ${payloadHash}, ${JSON.stringify({
              ...response,
              session_id: sessionId,
            })}::jsonb, NOW())
          ON CONFLICT (idempotency_key) DO NOTHING
          RETURNING id
        \`;

        if (!eventRows.length) {
          duplicate.push(id);
          continue;
        }

        await tx\`
          INSERT INTO diagnostic_responses
            (session_id, question_id, selected_option, response_time_ms, synced_at, sync_event_id)
          VALUES
            (${sessionId}, ${questionId}, ${selectedOption}, ${responseTimeMs}, NOW(), ${eventRows[0].id})
          ON CONFLICT (session_id, question_id) DO NOTHING
        \`;

        await tx\`
          INSERT INTO audit_logs
            (actor_type, actor_id, action, entity_type, entity_id, school_id)
          VALUES
            ('teacher', ${teacher.uid}, 'create', 'diagnostic_response', ${id}, ${teacher.schoolId})
        \`;

        accepted.push(id);
      }
    });

    return Response.json({
      accepted,
      duplicate,
      rejected,
      idempotency_ids: [...accepted, ...duplicate],
      server_time: new Date().toISOString(),
    });
  } catch (error) {
    if (error instanceof Response) return error;
    return Response.json({error:"sync_failed"}, {status:500});
  }
}
