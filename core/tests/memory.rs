use attune_core::memory::SemanticMemory;
#[test] fn retrieves_relevant_memory(){let mut m=SemanticMemory::default();m.insert("1","university economics assignment",0,1.0,0.0);m.insert("2","weather forecast and rain",0,0.2,0.0);let r=m.retrieve("economics assignment",0,1);assert_eq!(r[0].id,"1");}
#[test] fn sensitivity_is_penalized(){let mut m=SemanticMemory::default();m.insert("safe","my study goal",0,1.0,0.0);m.insert("sensitive","my study goal",0,1.0,1.0);let r=m.retrieve("study goal",0,2);assert_eq!(r[0].id,"safe");}
