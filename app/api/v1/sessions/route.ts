import { requireConsent, requireStudentAccess, requireTeacher } from "../../../../../lib/server/auth";
import { db } from "../../../../../lib/server/db";
import { issueSessionToken, hashSessionToken, sessionExpiry } from "../../../../../lib/server/session-token";
import { QUESTIONS } from "../../../../../lib/questions";
import { randomUUID } from "node:crypto";

export const runtime = "nodejs";

type Input = {
  student_id?: unknown;
  class?: unknown;
  subject?: unknown;
  language?: unknown;
  device_type?: unknown;
};

function selectQuestions(classNo: number, subject: "maths" | "science") {
  const pool = QUESTIONS.filter((q) => q.class === classNo && q.subject === subject && q.reviewStatus === "teacher-approved");
  const byTopic = new Map<string, typeof pool>();
  for (const q of pool) {
    const list = byTopic.get(q.topic) ?? [];
    list.push(q);
    byTopic.set(q.topic, list);
  }

  const selected = [...byTopic.values()].flatMap((items) => items.slice(0, 2));
  const deduped = [...new Map(selected.map((q) => [q.id, q])).values()];
  return deduped.slice(0, 15);
}

export async function POST(request: Request) {
  try {
    const teacher = await requireTeacher(request);
    const body = (await request.json()) as Input;
    const studentId = Number(body.student_id);
    const classNo = Number(body.class);
    const subject = body.subject === "science" ? "science" : body.subject === "maths" ? "maths" : "";
    const language = body.language === "en" ? "en" : "or";
    const deviceType = typeof body.device_type === "string" && body.device_type.length <= 32 ? body.device_type : "browser";

    if (!Number.isInteger(studentId) || studentId <= 0 || ![8, 9].includes(classNo) || !subject) {
      return Response.json({error:"invalid_session_request"}, {status:400});
    }

    const student = await requireStudentAccess(teacher.teacherId, studentId);
    if (student.classNo !== classNo) return Response.json({error:"class_mismatch"}, {status:400});
    await requireConsent(studentId);

    const selected = selectQuestions(classNo, subject);
    if (selected.length < 10) return Response.json({error:"insufficient_question_bank"}, {status:503});

    const clientSessionId = randomUUID();
    const token = issueSessionToken();
    const tokenHash = hashSessionToken(token);
    const expiresAt = sessionExpiry();

    const rows = await db()`
      INSERT INTO diagnostic_sessions
        (student_id, started_at, language, device_type, sync_status, client_session_id,
         session_token_hash, issued_at, expires_at, class_no, subject, question_ids)
      VALUES
        (${studentId}, NOW(), ${language}, ${deviceType}, 'pending', ${clientSessionId},
         ${tokenHash}, NOW(), ${expiresAt}, ${classNo}, ${subject}, ${JSON.stringify(selected.map((q) => q.id))}::jsonb)
      RETURNING id, client_session_id, expires_at
    `;

    const session = rows[0];
    await db()`
      INSERT INTO audit_logs
        (actor_type, actor_id, action, entity_type, entity_id, school_id)
      VALUES
        ('teacher', ${teacher.uid}, 'create', 'diagnostic_session', ${String(session.id)}, ${teacher.schoolId})
    `;

    return Response.json({
      session_id: Number(session.id),
      client_session_id: session.client_session_id,
      session_token: token,
      expires_at: session.expires_at,
      class: classNo,
      subject,
      language,
      questions: selected.map(({answerIndex: _answerIndex, ...question}) => question),
    }, {status:201});
  } catch (error) {
    if (error instanceof Response) return error;
    return Response.json({error:"session_creation_failed"}, {status:500});
  }
}
