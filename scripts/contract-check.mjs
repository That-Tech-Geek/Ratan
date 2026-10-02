import fs from "node:fs";
const required=["app/page.tsx","app/layout.tsx","app/api/v1/health/route.ts","app/api/v1/diagnostic/questions/route.ts","app/api/v1/likert/items/route.ts","app/api/v1/sync/batch/route.ts","lib/offline.ts","lib/server/db.ts","public/manifest.webmanifest"];
for(const f of required)if(!fs.existsSync(f))throw new Error("Missing Vercel app component: "+f);
const page=fs.readFileSync("app/page.tsx","utf8"),sync=fs.readFileSync("app/api/v1/sync/batch/route.ts","utf8");
if(!page.includes("/api/v1/diagnostic/questions"))throw new Error("Missing diagnostic client contract");
if(!sync.includes("audit_logs"))throw new Error("Missing persistence contract");
console.log("gyaan-saathi Vercel contract: PASS");