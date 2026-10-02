import postgres from "postgres";
if(!process.env.DATABASE_URL)throw new Error("DATABASE_URL is required");
const sql=postgres(process.env.DATABASE_URL,{max:1,prepare:false});
const tables=["schema_migrations","schools","students","teachers","consents","diagnostic_sessions","diagnostic_responses","likert_sessions","likert_responses","learning_preferences","sync_events","audit_logs","deletion_queue","recheck_sessions"];
const columns={diagnostic_sessions:["client_session_id","session_token_hash","issued_at","expires_at","class_no","subject","question_ids"],diagnostic_responses:["sync_event_id"],likert_responses:["response_event_id"],consents:["withdrawn_at","evidence_ref"],teachers:["firebase_uid","school_id","active"],learning_preferences:["top_preference","is_mixed"]};
try{
 const actual=new Set((await sql`SELECT table_name FROM information_schema.tables WHERE table_schema='public'`).map(r=>r.table_name));
 for(const t of tables)if(!actual.has(t))throw new Error("Missing table: "+t);
 for(const [t,cs] of Object.entries(columns)){const rows=await sql`SELECT column_name FROM information_schema.columns WHERE table_schema='public' AND table_name=${t}`;const s=new Set(rows.map(r=>r.column_name));for(const c of cs)if(!s.has(c))throw new Error("Missing column: "+t+"."+c);}
 const versions=await sql`SELECT version FROM schema_migrations ORDER BY version`.then(rs=>rs.map(r=>r.version));const expected=["001_initial","002_sessions","003_auth_tenancy","004_sync_materialization","005_diagnostic_selection","006_likert_preferences","007_consent_deletion","008_rechecks","009_retention_indexes"];if(JSON.stringify(versions)!==JSON.stringify(expected))throw new Error("Unexpected migrations: "+versions.join(","));
 console.log("database schema verification: PASS");
}finally{await sql.end({timeout:1});}