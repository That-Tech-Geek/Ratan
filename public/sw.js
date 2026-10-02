const VERSION = "2026-10-03";
const SHELL_CACHE = `gyaan-saathi-shell-${VERSION}`;
const RUNTIME_CACHE = `gyaan-saathi-runtime-${VERSION}`;
const DATA_CACHE = `gyaan-saathi-data-${VERSION}`;
const MAX_RUNTIME_ENTRIES = 80;
const SHELL = ["/", "/manifest.webmanifest"];

self.addEventListener("install", (event) => {
  event.waitUntil(caches.open(SHELL_CACHE).then((cache) => cache.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (event) => {
  event.waitUntil(caches.keys().then((keys) => Promise.all(
    keys.filter((key) =>
      (key.startsWith("gyaan-saathi-shell-") || key.startsWith("gyaan-saathi-runtime-") || key.startsWith("gyaan-saathi-data-")) &&
      ![SHELL_CACHE, RUNTIME_CACHE, DATA_CACHE].includes(key),
    ).map((key) => caches.delete(key)),
  )).then(() => self.clients.claim()));
});

async function trimCache(cacheName, maxEntries) {
  const cache = await caches.open(cacheName);
  const requests = await cache.keys();
  await Promise.all(requests.slice(0, Math.max(0, requests.length - maxEntries)).map((request) => cache.delete(request)));
}

async function networkFirst(request, cacheName) {
  const cache = await caches.open(cacheName);
  try {
    const response = await fetch(request);
    if (response.ok && response.type === "basic") {
      await cache.put(request, response.clone());
      await trimCache(cacheName, MAX_RUNTIME_ENTRIES);
    }
    return response;
  } catch {
    const cached = await cache.match(request);
    if (cached) return cached;
    throw new Error("offline");
  }
}

self.addEventListener("fetch", (event) => {
  const request = event.request;
  if (request.method !== "GET") return;
  const url = new URL(request.url);
  if (url.origin !== self.location.origin) return;

  if (url.pathname === "/api/v1/diagnostic/questions") {
    event.respondWith(networkFirst(request, DATA_CACHE).catch(() =>
      new Response(JSON.stringify({ version: "offline", questions: [] }), {
        status: 503, headers: { "content-type": "application/json" },
      }),
    ));
    return;
  }

  if (request.mode === "navigate") {
    event.respondWith(networkFirst(request, SHELL_CACHE).catch(() =>
      caches.match("/").then((cached) => cached || new Response("Offline", { status: 503 })),
    ));
    return;
  }

  if (url.pathname.startsWith("/_next/static/") || url.pathname === "/manifest.webmanifest") {
    event.respondWith(caches.match(request).then(async (cached) => {
      if (cached) return cached;
      const response = await fetch(request);
      if (response.ok && response.type === "basic") {
        const cache = await caches.open(RUNTIME_CACHE);
        await cache.put(request, response.clone());
        await trimCache(RUNTIME_CACHE, MAX_RUNTIME_ENTRIES);
      }
      return response;
    }));
  }
});

self.addEventListener("sync", (event) => {
  if (event.tag !== "gyaan-saathi-sync") return;
  event.waitUntil(self.clients.matchAll({ type: "window", includeUncontrolled: true })
    .then((clients) => clients.forEach((client) => client.postMessage({ type: "SYNC_REQUESTED" }))));
});
