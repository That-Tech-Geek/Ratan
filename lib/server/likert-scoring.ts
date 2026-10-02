export type LikertResponse={itemId:string;score:number};
const styles=["visual","auditory","reading","kinesthetic"] as const;
export function computePreference(responses:LikertResponse[]){
  const scores={visual:0,auditory:0,reading:0,kinesthetic:0};
  for(const r of responses){
    const n=Number(r.itemId.replace(/\D/g,""))||0;
    const style=styles[(n-1)%4];
    scores[style]+=r.score;
  }
  const sorted=Object.entries(scores).sort((a,b)=>b[1]-a[1]);
  const [top,second]=sorted;
  return {
    visual_score:scores.visual,auditory_score:scores.auditory,
    reading_writing_score:scores.reading,kinesthetic_score:scores.kinesthetic,
    top_preference:top[0],second_preference:second[0],
    is_mixed:top[1]-second[1]<=3
  };
}
