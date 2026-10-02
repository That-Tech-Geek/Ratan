import fs from "node:fs";

const required = [
  "app/page.tsx","app/layout.tsx","app/api/v1/health/route.ts",
  "app/api/v1/diagnostic/questions/route.ts","app/api/v1/likert/items/route.ts",
  "app/api/v1/sync/batch/route.ts","lib/offline.ts","lib/browser-cache.ts",
  "lib/server/db.ts","public/manifest.webmanifest","public/sw.js",
];

for (const file of required) {
  if (!fs.existsSync(file)) throw new Error("Missing Vercel app component: " + file);
}

const page = fs.readFileSync("app/page.tsx", "utf8");
const sw = fs.readFileSync("public/sw.js", "utf8");
const sync = fs.readFileSync("app/api/v1/sync/batch/route.ts", "utf8");
const questions = fs.readFileSync("lib/questions.ts", "utf8");

if (!page.includes("/api/v1/diagnostic/questions")) throw new Error("Missing diagnostic client contract");
if (!page.includes("enqueue")) throw new Error("Missing offline queue contract");
if (!sw.includes("caches.open")) throw new Error("Missing Cache Storage contract");
if (!sw.includes("gyaan-saathi-sync")) throw new Error("Missing background sync contract");
if (!sync.includes("audit_logs") || !sync.includes("sync_events")) throw new Error("Missing idempotent persistence contract");
if (!questions.includes('class:8') || !questions.includes('class:9') || !questions.includes('subject:"science"')) throw new Error("Incomplete maths/science question bank");
if (!questions.includes('reviewStatus:"teacher-approved"')) throw new Error("Question review gate missing");
if (!fs.readFileSync("lib/offline.ts","utf8").includes("this.version(2)")) throw new Error("IndexedDB schema version missing");

console.log("gyaan-saathi production offline/cache contract: PASS");
