type Bucket={count:number;resetAt:number};
const buckets=new Map<string,Bucket>();
export function rateLimit(key:string,limit:number,windowMs:number){
  const now=Date.now();
  let b=buckets.get(key);
  if(!b||now>=b.resetAt){b={count:0,resetAt:now+windowMs};buckets.set(key,b);}
  b.count++;
  if(b.count>limit) throw new Response("Too many requests",{status:429});
}
