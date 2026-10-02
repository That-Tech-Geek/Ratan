import {QUESTIONS} from "../../../../../lib/questions";
export const runtime="nodejs";
const SUBJECTS=new Set(["maths","science"]);
const CLASSES=new Set([8,9]);
export async function GET(request:Request){
 const u=new URL(request.url); const cls=Number(u.searchParams.get("class")||8); const subject=u.searchParams.get("subject")||"maths";
 if(!CLASSES.has(cls)||!SUBJECTS.has(subject))return Response.json({error:"invalid_diagnostic_scope"},{status:400});
 const questions=QUESTIONS.filter(q=>q.class===cls&&q.subject===subject&&q.reviewStatus==="teacher-approved").map(({answerIndex, ...publicQuestion})=>publicQuestion);
 return Response.json({version:"2",class:cls,subject,language:"or",question_count:questions.length,questions},{headers:{"cache-control":"public,max-age=0,must-revalidate"}});
}