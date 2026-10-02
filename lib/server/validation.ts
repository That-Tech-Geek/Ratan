import {z} from "zod";
const UUID=z.string().uuid();
const DiagnosticResponse=z.object({id:UUID,entity:z.literal("diagnostic_response"),action:z.literal("create"),payload:z.object({question_id:z.string().min(1).max(64),selected_option:z.string().min(1).max(128),response_time_ms:z.number().int().nonnegative().max(3600000)})});
export function parseDiagnosticResponse(item:unknown){
 const parsed=DiagnosticResponse.safeParse(item);
 if(!parsed.success)return {ok:false as const,reason:"invalid_schema"};
 const p=parsed.data.payload;
 return {ok:true as const,id:parsed.data.id,questionId:p.question_id,selectedOption:p.selected_option,responseTimeMs:p.response_time_ms,payload:p};
}