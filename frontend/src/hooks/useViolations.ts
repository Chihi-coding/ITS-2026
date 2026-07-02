import { useEffect, useState, useCallback } from "react";
import { supabase } from "../lib/supabase";
import type { Violation } from "../types/violation";

/**
 * Hook that fetches violations from Supabase and subscribes to real-time
 * INSERT events so the table updates instantly when the AI module reports
 * a new violation.
 */
export function useViolations() {
  const [violations, setViolations] = useState<Violation[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [realtimeConnected, setRealtimeConnected] = useState(false);

  /* ── Initial fetch ──────────────────────────────────────── */
  const fetchViolations = useCallback(async () => {
    try {
      setLoading(true);
      const { data, error: fetchError } = await supabase
        .from("violations")
        .select("*")
        .order("id", { ascending: false });

      if (fetchError) throw fetchError;
      setViolations(data ?? []);
      setError(null);
    } catch (err: unknown) {
      const msg =
        err instanceof Error
          ? err.message
          : typeof err === "object" && err !== null && "message" in err
            ? String((err as { message: unknown }).message)
            : JSON.stringify(err);
      setError(msg);
      console.error("[useViolations] fetch error:", err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchViolations();
  }, [fetchViolations]);

  /* ── Real-time subscription ─────────────────────────────── */
  useEffect(() => {
    const channel = supabase
      .channel("violations-realtime")
      .on(
        "postgres_changes",
        {
          event: "INSERT",
          schema: "public",
          table: "violations",
        },
        (payload) => {
          console.log("[Realtime] New violation:", payload.new);
          setViolations((prev) => [payload.new as Violation, ...prev]);
        },
      )
      .on(
        "postgres_changes",
        {
          event: "UPDATE",
          schema: "public",
          table: "violations",
        },
        (payload) => {
          console.log("[Realtime] Updated violation:", payload.new);
          setViolations((prev) =>
            prev.map((v) =>
              v.id === (payload.new as Violation).id
                ? (payload.new as Violation)
                : v,
            ),
          );
        },
      )
      .subscribe((status) => {
        console.log("[Realtime] channel status:", status);
        setRealtimeConnected(status === "SUBSCRIBED");
      });

    return () => {
      supabase.removeChannel(channel);
    };
  }, []);

  return { violations, loading, error, realtimeConnected, refetch: fetchViolations };
}
