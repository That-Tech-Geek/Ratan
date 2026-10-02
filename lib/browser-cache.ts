const CACHE_NAME = "gyaan-saathi-data-v1";
const QUESTIONS_TTL_MS = 24 * 60 * 60 * 1000;

type CachedQuestions = {
  version: string;
  class: number;
  subject: string;
  questions: Array<{ id: string; prompt: string; options: string[] }>;
  cachedAt: number;
};

function cacheKey(classNo: number, subject: string) {
  return new Request(`/api/v1/diagnostic/questions?class=${encodeURIComponent(classNo)}&subject=${encodeURIComponent(subject)}`);
}

export async function getCachedQuestions(classNo: number, subject: string) {
  if (!("caches" in globalThis)) return null;
  const cache = await caches.open(CACHE_NAME);
  const response = await cache.match(cacheKey(classNo, subject));
  if (!response) return null;

  try {
    const payload = (await response.json()) as CachedQuestions;
    if (!payload?.cachedAt || Date.now() - payload.cachedAt > QUESTIONS_TTL_MS) return null;
    return payload;
  } catch {
    await cache.delete(cacheKey(classNo, subject));
    return null;
  }
}

export async function putCachedQuestions(
  payload: Omit<CachedQuestions, "cachedAt">,
) {
  if (!("caches" in globalThis)) return;
  const cache = await caches.open(CACHE_NAME);
  const body: CachedQuestions = { ...payload, cachedAt: Date.now() };
  await cache.put(
    cacheKey(payload.class, payload.subject),
    new Response(JSON.stringify(body), {
      headers: { "content-type": "application/json", "cache-control": "no-store" },
    }),
  );
}

export async function clearBrowserDataCache() {
  if (!("caches" in globalThis)) return;
  await caches.delete(CACHE_NAME);
}
