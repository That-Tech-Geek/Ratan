export type DiagnosticResponseInput={id:string;entity:string;action:string;payload:unknown};
const UUID=/^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
export function isUuid(v:unknown):v is string{return typeof v==="string"&&UUID.test(v);}
export function parseDiagnosticResponse(item:unknown){
  if(!item||typeof item!=="object") return {ok:false as const,reason:"invalid_item"};
  const x=item as Record<string,unknown>, p=x.payload;
  if(!isUuid(x.id)) return {ok:false as const,reason:"invalid_id"};
  if(x.entity!=="diagnostic_response"||x.action!=="create") return {ok:false as const,reason:"unsupported_entity"};
  if(!p||typeof p!=="object") return {ok:false as const,reason:"invalid_payload"};
  const r=p as Record<string,unknown>;
  if(typeof r.question_id!=="string"||r.question_id.length<1||r.question_id.length>64) return {ok:false as const,reason:"invalid_question_id"};
  if(typeof r.selected_option!=="string"||r.selected_option.length>128) return {ok:false as const,reason:"invalid_selected_option"};
  const t=Number(r.response_time_ms);
  if(!Number.isInteger(t)||t<0||t>3600000) return {ok:false as const,reason:"invalid_response_time"};
  return {ok:true as const,id:x.id,questionId:r.question_id,selectedOption:r.selected_option,responseTimeMs:t,payload:r};
}
