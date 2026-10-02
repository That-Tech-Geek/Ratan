"use client";

import { createClient, type AuthChangeEvent, type Session, type SupabaseClient } from "@supabase/supabase-js";

let client: SupabaseClient | null = null;

function getSupabase() {
  if (typeof window === "undefined") return null;
  const url = process.env.SUPABASE_URL;
  const key = process.env.SUPABASE_PUBLISHABLE_KEY;
  if (!url || !key) throw new Error("supabase_config_missing");
  client ??= createClient(url, key, {
    auth: { persistSession: true, autoRefreshToken: true, detectSessionInUrl: true },
  });
  return client;
}

export function subscribeAuth(callback: (session: Session | null) => void) {
  const supabase = getSupabase();
  if (!supabase) return () => {};
  const { data } = supabase.auth.onAuthStateChange((_event: AuthChangeEvent, session) => callback(session));
  void supabase.auth.getSession().then(({ data: current }) => callback(current.session));
  return () => data.subscription.unsubscribe();
}

export async function requestOtp(phone: string) {
  const supabase = getSupabase();
  if (!supabase) throw new Error("supabase_unavailable");
  return supabase.auth.signInWithOtp({ phone });
}

export async function verifyOtp(phone: string, token: string) {
  const supabase = getSupabase();
  if (!supabase) throw new Error("supabase_unavailable");
  return supabase.auth.verifyOtp({ phone, token, type: "sms" });
}

export async function currentIdToken() {
  const supabase = getSupabase();
  if (!supabase) return null;
  const { data } = await supabase.auth.getSession();
  return data.session?.access_token ?? null;
}

export async function signOutUser() {
  const supabase = getSupabase();
  if (supabase) await supabase.auth.signOut();
}
