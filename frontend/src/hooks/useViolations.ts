import { useEffect, useState, useCallback, useRef } from "react";
import { supabase } from "../lib/supabase";
import type { Violation } from "../types/violation";

/** Polling interval in ms — lightweight fallback when Realtime is flaky. */
const POLL_INTERVAL_MS = 5_000;

/**
 * Hook that fetches violations from Supabase, subscribes to real-time
 * INSERT/UPDATE events, **and** runs a lightweight polling fallback so the
 * table always stays current even when Supabase Realtime is misconfigured
 * (e.g. replication not enabled on the table, RLS blocking anon reads, etc.).
 */
export function useViolations() {
  const [violations, setViolations] = useState<Violation[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [realtimeConnected, setRealtimeConnected] = useState(false);

  // Track the highest known ID so the polling query is cheap (fetch only new rows)
  const maxIdRef = useRef<number>(0);

  /* ── Initial fetch ──────────────────────────────────────── */
  const fetchViolations = useCallback(async () => {
    try {
      setLoading(true);
      const { data, error: fetchError } = await supabase
        .from("violations")
        .select("*")
        .order("id", { ascending: false });

      if (fetchError) throw fetchError;

      const rows = data ?? [];
      setViolations(rows);
      setError(null);

      // Seed the max-id watermark
      if (rows.length > 0) {
        maxIdRef.current = rows[0].id;
      }
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
          const incoming = payload.new as Violation;

          setViolations((prev) => {
            // Deduplicate — the polling fallback may have already added this row
            if (prev.some((v) => v.id === incoming.id)) return prev;
            return [incoming, ...prev];
          });

          // Advance watermark so polling doesn't re-fetch this row
          if (incoming.id > maxIdRef.current) {
            maxIdRef.current = incoming.id;
          }
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

  /* ── Polling fallback ───────────────────────────────────── */
  useEffect(() => {
    const poll = async () => {
      try {
        // Only fetch rows newer than our watermark — very lightweight query
        const { data, error: pollError } = await supabase
          .from("violations")
          .select("*")
          .gt("id", maxIdRef.current)
          .order("id", { ascending: false });

        if (pollError) {
          console.warn("[useViolations] poll error:", pollError.message);
          return;
        }

        if (data && data.length > 0) {
          console.log(`[useViolations] poll found ${data.length} new row(s)`);

          setViolations((prev) => {
            // Deduplicate against existing state (Realtime may have delivered some already)
            const existingIds = new Set(prev.map((v) => v.id));
            const genuinelyNew = data.filter((v) => !existingIds.has(v.id));
            if (genuinelyNew.length === 0) return prev;

            console.log(
              `[useViolations] prepending ${genuinelyNew.length} new violation(s)`,
            );
            return [...genuinelyNew, ...prev];
          });

          // Advance watermark
          maxIdRef.current = data[0].id;
        }
      } catch (err) {
        console.warn("[useViolations] poll exception:", err);
      }
    };

    const id = setInterval(poll, POLL_INTERVAL_MS);
    return () => clearInterval(id);
  }, []);

  return {
    violations,
    loading,
    error,
    realtimeConnected,
    refetch: fetchViolations,
  };
}
