# Privacy-native runtime

`EncryptedVault` provides authenticated AES-256-GCM encryption for local records. The 32-byte master key is supplied by the host and must come from an OS secure keystore/keychain in a production mobile build. The core never persists the master key.

Deletion removes the encrypted record from the vault. Export should be implemented at the host boundary with explicit user consent. Network telemetry is not part of the core runtime.
