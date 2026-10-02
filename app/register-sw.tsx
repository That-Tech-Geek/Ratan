"use client";

import { useEffect } from "react";

export default function RegisterSW() {
  useEffect(() => {
    if (!("serviceWorker" in navigator)) return;
    let mounted = true;

    void navigator.serviceWorker.register("/sw.js", { scope: "/" })
      .then(async (registration) => {
        if (!mounted) return;
        await registration.update();
        if ("sync" in registration) {
          try { await (registration as ServiceWorkerRegistration & { sync: { register(tag: string): Promise<void> } }).sync.register("gyaan-saathi-sync"); } catch {}
        }
      })
      .catch(() => undefined);

    return () => { mounted = false; };
  }, []);

  return null;
}
