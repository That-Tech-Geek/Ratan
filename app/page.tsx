"use client";

import { useEffect, useMemo, useState } from "react";
import { db, enqueue, getDiagnosticSession } from "../lib/offline";
import { getCachedQuestions, putCachedQuestions } from "../lib/browser-cache";

type Question = { id: string; prompt: string; options: string[] };
type QuestionsPayload = { version: string; class: number; subject: string; questions: Question[] };

const CLASS_NO = 8;
const SUBJECT = "maths";
const BATCH_SIZE = 20;

async function loadQuestions() {
  const cached = await getCachedQuestions(CLASS_NO, SUBJECT);
  if (cached?.questions?.length) return { payload: cached as QuestionsPayload, source: "cache" as const };

  const response = await fetch(`/api/v1/diagnostic/questions?class=${CLASS_NO}&subject=${SUBJECT}`, {
    cache: "no-store",
  });
  if (!response.ok) throw new Error("questions_unavailable");
  const payload = (await response.json()) as QuestionsPayload;
  await putCachedQuestions(payload);
  return { payload, source: "network" as const };
}

async function flushQueue() {
  if (!navigator.onLine) return 0;
  const session = await getDiagnosticSession();
  if (!session || new Date(session.expiresAt).getTime() <= Date.now()) return 0;
  await db.queue.where("status").equals("syncing").modify({ status: "pending" });
  const pending = (await db.queue.where("status").anyOf("pending", "failed").sortBy("createdAt")).filter((item) => item.retryCount < 5);
  let accepted = 0;

  for (let offset = 0; offset < pending.length; offset += BATCH_SIZE) {
    const batch = pending.slice(offset, offset + BATCH_SIZE);
    await db.transaction("rw", db.queue, async () => {
      await Promise.all(batch.map((item) => db.queue.update(item.id, { status: "syncing" })));
    });

    try {
      const response = await fetch("/api/v1/sync/batch", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ session_id: session.sessionId, session_token: session.sessionToken, items: batch }),
      });
      if (!response.ok) throw new Error("sync_failed");
      const result = (await response.json()) as { idempotency_ids?: string[] };
      const ids = new Set(result.idempotency_ids ?? []);
      await db.transaction("rw", db.queue, async () => {
        for (const item of batch) {
          if (ids.has(item.id)) {
            await db.queue.delete(item.id);
            accepted += 1;
          } else {
            await db.queue.update(item.id, {
              status: "failed",
              retryCount: item.retryCount + 1,
            });
          }
        }
      });
    } catch {
      await db.transaction("rw", db.queue, async () => {
        for (const item of batch) {
          await db.queue.update(item.id, {
            status: "pending",
            retryCount: item.retryCount + 1,
          });
        }
      });
      break;
    }
  }

  return accepted;
}

export default function Home() {
  const [questions, setQuestions] = useState<Question[]>([]);
  const [index, setIndex] = useState(0);
  const [answers, setAnswers] = useState<Record<string, string>>({});
  const [online, setOnline] = useState(true);
  const [queued, setQueued] = useState(0);
  const [loading, setLoading] = useState(true);
  const [source, setSource] = useState<"network" | "cache" | "none">("none");
  const [done, setDone] = useState(false);
  const [startedAt] = useState(() => Date.now());


  useEffect(() => {
    let mounted = true;

    const refresh = async () => {
      if (!mounted) return;
      setOnline(navigator.onLine);
      setQueued(await db.queue.count());
    };

    const sync = async () => {
      await flushQueue();
      await refresh();
    };

    const onOnline = () => void sync();
    const onOffline = () => void refresh();
    const onMessage = (event: MessageEvent) => {
      if (event.data?.type === "SYNC_REQUESTED") void sync();
    };

    addEventListener("online", onOnline);
    addEventListener("offline", onOffline);
    navigator.serviceWorker?.addEventListener("message", onMessage);

    void (async () => {
      try {
        const result = await loadQuestions();
        if (mounted) {
          setQuestions(result.payload.questions);
          setSource(result.source);
        }
      } catch {
        if (mounted) setSource("none");
      } finally {
        if (mounted) setLoading(false);
      }
      await sync();
    })();

    return () => {
      mounted = false;
      removeEventListener("online", onOnline);
      removeEventListener("offline", onOffline);
      navigator.serviceWorker?.removeEventListener("message", onMessage);
    };
  }, []);

  const q = questions[index];
  const progress = useMemo(
    () => questions.length ? Math.round(((index + (done ? 1 : 0)) / questions.length) * 100) : 0,
    [index, questions.length, done],
  );

  const choose = async (option: string) => {
    if (!q) return;
    setAnswers((current) => ({ ...current, [q.id]: option }));
    await enqueue({
      entity: "diagnostic_response",
      action: "create",
      payload: { question_id: q.id, selected_option: option, response_time_ms: Math.max(0, Date.now() - startedAt) },
    });
    setQueued(await db.queue.count());

    if (navigator.onLine) {
      await flushQueue();
      setQueued(await db.queue.count());
    }

    try {
      const registration = await navigator.serviceWorker?.ready;
      if (registration && "sync" in registration) {
        try { await (registration as ServiceWorkerRegistration & { sync: { register(tag: string): Promise<void> } }).sync.register("gyaan-saathi-sync"); } catch {}
      }
    } catch {}
  };

  const next = () => {
    if (index + 1 < questions.length) setIndex(index + 1);
    else setDone(true);
  };

  return (
    <main className="shell">
      <header className="topbar">
        <div><div className="brand">Gyaan Saathi</div><div className="muted">Learning diagnostics</div></div>
        <span className="badge">{online ? "Online" : "Offline"} · {queued} queued</span>
      </header>

      <section className="hero">
        <h1>Learn where you are.</h1>
        <p>Short diagnostics that keep working on low-end Android devices and unreliable 2G/3G connections. Responses are saved locally first and synchronized when connectivity returns.</p>
      </section>

      <div className="grid">
        <article className="card"><h3>Diagnostic</h3><p className="muted">Class 8 mathematics starter assessment.</p></article>
        <article className="card"><h3>Offline-first</h3><p className="muted">Cache Storage keeps public learning content available; IndexedDB is the durable response queue.</p></article>
        <article className="card"><h3>Teacher-ready</h3><p className="muted">The same response contract can feed reports and re-checks after server acknowledgement.</p></article>
      </div>

      <section className="card">
        {loading ? <p>Loading questions…</p> :
        done ? <><h2>Diagnostic saved</h2><p className="muted">{Object.keys(answers).length} response(s) are stored locally and synchronized when available.</p><button className="primary" onClick={() => location.reload()}>Start again</button></> :
        q ? <><div className="progress"><span style={{ width: progress + "%" }} /></div><p className="muted">Question {index + 1} of {questions.length} · {source === "cache" ? "cached content" : "fresh content"}</p><div className="question">{q.prompt}</div><div className="options">{q.options.map((option) => <button key={option} className={"option " + (answers[q.id] === option ? "selected" : "")} onClick={() => void choose(option)}>{option}</button>)}</div><div className="actions" style={{ marginTop: 18 }}><button className="primary" disabled={!answers[q.id]} onClick={next}>{index + 1 === questions.length ? "Finish" : "Next"}</button></div></> :
        <><p>No questions are available locally.</p><p className="muted">Reconnect once to prime the browser cache, then the diagnostic can continue through connectivity loss.</p></>}
      </section>

      <footer className="footer">Gyaan Saathi · Offline-first education infrastructure</footer>
    </main>
  );
}
