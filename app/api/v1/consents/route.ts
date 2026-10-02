import {requireStudentAccess,requireTeacher} from "../../../../lib/server/auth";
import {db} from "../../../../lib/server/db";
export const runtime="nodejs";
export async function POST(request:Request){
 try{const teacher=await requireTeacher(request);const b=await request.json();const studentId=Number(b.student_id);await requireStudentAccess(teacher.teacherId,studentId);
 const granted=Boolean(b.granted), evidenceRef=typeof b.evidence_ref==="string"?b.evidence_ref.trim().slice(0,2000):"";
 await db()`INSERT INTO consents(student_id,consent_type,consent_version,granted,evidence_ref) VALUES(${studentId},'diagnostic','v1',${granted},${evidenceRef}) ON CONFLICT(student_id,consent_type,consent_version) DO UPDATE SET granted=EXCLUDED.granted,evidence_ref=EXCLUDED.evidence_ref,recorded_at=NOW(),withdrawn_at=NULL`;
 if(!granted) await db()`INSERT INTO deletion_queue(student_id,reason) VALUES(${studentId},'consent_withdrawal')`;
 await db()`INSERT INTO audit_logs(actor_type,actor_id,action,entity_type,entity_id,school_id) VALUES('teacher',${teacher.uid},${granted?"grant":"withdraw"},'consent',${String(studentId)},${teacher.schoolId})`;
 return Response.json({ok:true});
 }catch(e){if(e instanceof Response)throw e;return Response.json({error:"consent_failed"},{status:500});}
}