import { cert, getApps, initializeApp } from "firebase-admin/app";
import { getAuth } from "firebase-admin/auth";
import { db } from "./db";

function firebaseAuth() {
  if (!getApps().length) {
    const raw = process.env.FIREBASE_SERVICE_ACCOUNT_JSON ?? process.env.CREDS;
    if (!raw) throw new Error("FIREBASE_SERVICE_ACCOUNT_JSON or CREDS is required");
    const credentials = JSON.parse(raw);
    initializeApp({ credential: cert(credentials) });
  }
  return getAuth();
}

export async function requireTeacher(request: Request) {
  const header = request.headers.get("authorization");
  if (!header?.startsWith("Bearer ")) throw new Response("Unauthorized", {status:401});
  try {
    const decoded = await firebaseAuth().verifyIdToken(header.slice(7), true);
    const rows = await db()\`
      SELECT id, school_id, role
      FROM teachers
      WHERE firebase_uid = ${decoded.uid} AND active = TRUE
      LIMIT 1
    \`;
    const teacher = rows[0];
    if (!teacher) throw new Response("Forbidden", {status:403});
    return {uid: decoded.uid, teacherId: Number(teacher.id), schoolId: Number(teacher.school_id), role: teacher.role as string};
  } catch (error) {
    if (error instanceof Response) throw error;
    throw new Response("Unauthorized", {status:401});
  }
}

export async function requireStudentAccess(teacherId: number, studentId: number) {
  const rows = await db()\`
    SELECT s.id, s.school_id, s.class_no
    FROM students s
    JOIN teachers t ON t.school_id = s.school_id
    WHERE t.id = ${teacherId} AND s.id = ${studentId} AND t.active = TRUE
    LIMIT 1
  \`;
  if (!rows[0]) throw new Response("Forbidden", {status:403});
  return {studentId:Number(rows[0].id), schoolId:Number(rows[0].school_id), classNo:Number(rows[0].class_no)};
}

export async function requireConsent(studentId: number) {
  const rows = await db()\`
    SELECT id FROM consents
    WHERE student_id = ${studentId}
      AND consent_type = 'diagnostic'
      AND granted = TRUE
      AND withdrawn_at IS NULL
    ORDER BY recorded_at DESC
    LIMIT 1
  \`;
  if (!rows[0]) throw new Response("Consent required", {status:403});
}
