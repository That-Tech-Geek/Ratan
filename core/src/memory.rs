//! Privacy-aware semantic memory. The default index is a deterministic cosine index so
//! retrieval is reproducible and does not require a heavyweight runtime. The trait boundary
//! permits an ANN backend without changing the runtime API.
use sha2::{Digest,Sha256};
use serde::{Serialize,Deserialize};
pub const EMBEDDING_DIMS:usize=64;
#[derive(Debug,Clone,Serialize,Deserialize)]
pub struct MemoryItem { pub id:String, pub text:String, pub timestamp:i64, pub importance:f32, pub sensitivity:f32, pub embedding:Vec<f32> }
fn embedding(text:&str)->Vec<f32>{let mut out=vec![0.0;EMBEDDING_DIMS];for token in text.to_lowercase().split_whitespace(){let mut h=Sha256::new();h.update(token.as_bytes());let d=h.finalize();let i=u64::from_le_bytes(d[0..8].try_into().unwrap()) as usize%EMBEDDING_DIMS;out[i]+=1.0;}let n=out.iter().map(|x|x*x).sum::<f32>().sqrt();if n>0.0{for x in &mut out{*x/=n;}}out}
fn cosine(a:&[f32],b:&[f32])->f32{a.iter().zip(b).map(|(x,y)|x*y).sum()}
#[derive(Default)] pub struct SemanticMemory{items:Vec<MemoryItem>}
impl SemanticMemory{pub fn insert(&mut self,id:impl Into<String>,text:impl Into<String>,timestamp:i64,importance:f32,sensitivity:f32){let text=text.into();self.items.push(MemoryItem{id:id.into(),embedding:embedding(&text),text,timestamp,importance:importance.clamp(0.0,1.0),sensitivity:sensitivity.clamp(0.0,1.0)});}pub fn retrieve(&self,query:&str,now:i64,k:usize)->Vec<MemoryItem>{let q=embedding(query);let mut scored:Vec<(f32,&MemoryItem)>=self.items.iter().map(|m|{let semantic=cosine(&q,&m.embedding);let age=((now-m.timestamp).unsigned_abs() as f32/86_400.0).min(3650.0);let temporal=1.0/(1.0+age/30.0);let score=0.65*semantic+0.2*m.importance+0.15*temporal-0.20*m.sensitivity;(score,m)}).collect();scored.sort_by(|a,b|b.0.total_cmp(&a.0));scored.into_iter().take(k).map(|(_,m)|m.clone()).collect()}pub fn len(&self)->usize{self.items.len()}}
