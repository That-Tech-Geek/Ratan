import fs from "node:fs";

const requiredFiles = [
  "backend/config/settings.py",
  "backend/config/urls.py",
  "backend/gyaan/models.py",
  "backend/gyaan/api.py",
  "src/db.ts",
  "src/sync.ts",
];

for (const file of requiredFiles) {
  if (!fs.existsSync(file)) {
    throw new Error("Missing data-flow component: " + file);
  }
}

const urls = fs.readFileSync("backend/config/urls.py", "utf8");
const client = fs.readFileSync("src/api.ts", "utf8");

for (const path of [
  "api/v1/sync/batch/",
  "api/v1/diagnostic/questions",
  "api/v1/likert/items",
]) {
  if (!urls.includes(path)) {
    throw new Error("Missing backend route contract: /" + path);
  }
}

for (const path of [
  "/api/v1/sync/batch/",
  "/api/v1/diagnostic/questions",
]) {
  if (!client.includes(path)) {
    throw new Error("Missing frontend API contract: " + path);
  }
}

console.log("data-flow contract: PASS");
