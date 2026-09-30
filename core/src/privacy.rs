//! Encrypted local vault primitive. Key management is deliberately outside this module:
//! callers must supply a 256-bit key from an OS secure keystore/keychain.
use aes_gcm::{Aes256Gcm,Nonce,KeyInit,aead::Aead};
use rand::RngCore;
use std::collections::BTreeMap;
use zeroize::Zeroizing;
#[derive(Debug,thiserror::Error)]pub enum VaultError{#[error("invalid key")]InvalidKey,#[error("encryption failed")]Encrypt,#[error("decryption failed")]Decrypt}
#[derive(Default)]pub struct EncryptedVault{items:BTreeMap<String,Vec<u8>>}
impl EncryptedVault{
 pub fn put(&mut self,key:&str,value:&[u8],master_key:&[u8;32])->Result<(),VaultError>{let cipher=Aes256Gcm::new_from_slice(master_key).map_err(|_|VaultError::InvalidKey)?;let mut nonce=[0u8;12];rand::thread_rng().fill_bytes(&mut nonce);let encrypted=cipher.encrypt(Nonce::from_slice(&nonce),value).map_err(|_|VaultError::Encrypt)?;let mut packed=nonce.to_vec();packed.extend(encrypted);self.items.insert(key.to_string(),packed);Ok(())}
 pub fn get(&self,key:&str,master_key:&[u8;32])->Result<Option<Zeroizing<Vec<u8>>>,VaultError>{let Some(packed)=self.items.get(key) else{return Ok(None)};if packed.len()<12{return Err(VaultError::Decrypt)}let cipher=Aes256Gcm::new_from_slice(master_key).map_err(|_|VaultError::InvalidKey)?;let plain=cipher.decrypt(Nonce::from_slice(&packed[..12]),&packed[12..]).map_err(|_|VaultError::Decrypt)?;Ok(Some(Zeroizing::new(plain)))}
 pub fn delete(&mut self,key:&str)->bool{self.items.remove(key).is_some()}
 pub fn keys(&self)->Vec<String>{self.items.keys().cloned().collect()}
}
#[cfg(test)]mod tests{use super::*;#[test]fn encrypt_roundtrip(){let mut v=EncryptedVault::default();let k=[7u8;32];v.put("x",b"secret",&k).unwrap();assert_eq!(&*v.get("x",&k).unwrap().unwrap(),b"secret");assert!(v.get("x",&[8u8;32]).is_err());assert!(v.delete("x"));}}
