import { createHash, randomBytes } from "node:crypto";

const SESSION_TTL_MS = 24 * 60 * 60 * 1000;

export function issueSessionToken() {
  return randomBytes(32).toString("base64url");
}

export function hashSessionToken(token: string) {
  return createHash("sha256").update(token).digest("hex");
}

export function sessionExpiry() {
  return new Date(Date.now() + SESSION_TTL_MS);
}
