import {
  ShieldAlert,
  Radio,
  RefreshCw,
  Clock,
  Car,
  ImageIcon,
  X,
  AlertTriangle,
  Send,
  CheckCircle2,
  XCircle,
} from "lucide-react";
import { useState, useCallback, useEffect, useRef } from "react";
import { useViolations } from "../hooks/useViolations";
import type { Violation } from "../types/violation";
import CCTVMonitor from "./CCTVMonitor";

/* ── Helpers ───────────────────────────────────────────────── */

function formatTimestamp(raw: string | null | undefined): string {
  if (!raw) return "—";
  try {
    return new Intl.DateTimeFormat("en-GB", {
      day: "2-digit",
      month: "short",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hour12: false,
      timeZone: "Asia/Ho_Chi_Minh",
    }).format(new Date(raw));
  } catch {
    return raw;
  }
}

function statusBadge(status: string) {
  const s = status.toLowerCase();
  if (s === "resolved" || s === "cleared")
    return "bg-emerald-500/15 text-emerald-400 ring-emerald-500/30";
  if (s === "reviewed")
    return "bg-sky-500/15 text-sky-400 ring-sky-500/30";
  return "bg-amber-500/15 text-amber-400 ring-amber-500/30";
}

function bestTimestamp(v: Violation): string {
  return formatTimestamp(v.detected_at ?? v.violation_started_at);
}

/* ── Toast System ──────────────────────────────────────────── */

type ToastKind = "success" | "error";

interface Toast {
  id: number;
  message: string;
  kind: ToastKind;
}

let _toastId = 0;

function ToastContainer({ toasts, dismiss }: { toasts: Toast[]; dismiss: (id: number) => void }) {
  return (
    <div className="fixed bottom-6 right-6 z-[60] flex flex-col gap-2 pointer-events-none">
      {toasts.map((t) => (
        <div
          key={t.id}
          className={`pointer-events-auto flex items-center gap-3 rounded-xl px-4 py-3 text-sm font-medium shadow-2xl backdrop-blur-sm ring-1 animate-slide-in ${
            t.kind === "success"
              ? "bg-emerald-950/90 text-emerald-300 ring-emerald-500/30"
              : "bg-red-950/90 text-red-300 ring-red-500/30"
          }`}
        >
          {t.kind === "success" ? (
            <CheckCircle2 size={16} className="shrink-0 text-emerald-400" />
          ) : (
            <XCircle size={16} className="shrink-0 text-red-400" />
          )}
          {t.message}
          <button
            onClick={() => dismiss(t.id)}
            className="ml-2 opacity-60 hover:opacity-100 transition"
          >
            <X size={14} />
          </button>
        </div>
      ))}
    </div>
  );
}

function useToast() {
  const [toasts, setToasts] = useState<Toast[]>([]);
  const timers = useRef<Map<number, ReturnType<typeof setTimeout>>>(new Map());

  const dismiss = useCallback((id: number) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
    const timer = timers.current.get(id);
    if (timer) {
      clearTimeout(timer);
      timers.current.delete(id);
    }
  }, []);

  const addToast = useCallback(
    (message: string, kind: ToastKind, duration = 4000) => {
      const id = ++_toastId;
      setToasts((prev) => [...prev, { id, message, kind }]);
      const timer = setTimeout(() => dismiss(id), duration);
      timers.current.set(id, timer);
    },
    [dismiss]
  );

  // Cleanup on unmount
  useEffect(() => {
    const t = timers.current;
    return () => t.forEach((timer) => clearTimeout(timer));
  }, []);

  return { toasts, addToast, dismiss };
}

/* ── Lightbox ──────────────────────────────────────────────── */

function Lightbox({ src, onClose }: { src: string; onClose: () => void }) {
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm animate-fade-in"
      onClick={onClose}
    >
      <button
        onClick={onClose}
        className="absolute top-4 right-4 rounded-full bg-gray-800/80 p-2 text-gray-300 transition hover:bg-gray-700 hover:text-white"
      >
        <X size={20} />
      </button>
      <img
        src={src}
        alt="Evidence"
        className="max-h-[85vh] max-w-[90vw] rounded-xl shadow-2xl object-contain"
        onClick={(e) => e.stopPropagation()}
      />
    </div>
  );
}

/* ── Stats Cards ───────────────────────────────────────────── */

function StatsBar({ violations }: { violations: Violation[] }) {
  const total = violations.length;
  const pending = violations.filter(
    (v) => v.status?.toLowerCase() === "pending"
  ).length;
  const today = violations.filter((v) => {
    const ts = v.detected_at ?? v.violation_started_at;
    if (!ts) return false;
    return new Date(ts).toDateString() === new Date().toDateString();
  }).length;

  const cards = [
    {
      label: "Total Violations",
      value: total,
      icon: ShieldAlert,
      gradient: "from-indigo-600 to-violet-600",
    },
    {
      label: "Pending Review",
      value: pending,
      icon: AlertTriangle,
      gradient: "from-amber-600 to-orange-600",
    },
    {
      label: "Today",
      value: today,
      icon: Clock,
      gradient: "from-emerald-600 to-teal-600",
    },
  ];

  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
      {cards.map((c) => (
        <div
          key={c.label}
          className="relative overflow-hidden rounded-xl border border-gray-800 bg-gray-900/70 p-5 backdrop-blur"
        >
          <div
            className={`absolute -top-6 -right-6 h-24 w-24 rounded-full bg-gradient-to-br ${c.gradient} opacity-20 blur-2xl`}
          />
          <div className="flex items-center gap-3">
            <div
              className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-gradient-to-br ${c.gradient}`}
            >
              <c.icon size={18} className="text-white" />
            </div>
            <div>
              <p className="text-2xl font-bold tracking-tight">{c.value}</p>
              <p className="text-xs text-gray-400">{c.label}</p>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

/* ── Send Alert Button ─────────────────────────────────────── */

function SendAlertButton({
  violation,
  onSuccess,
  onError,
}: {
  violation: Violation;
  onSuccess: (msg: string) => void;
  onError: (msg: string) => void;
}) {
  const [loading, setLoading] = useState(false);
  const [sent, setSent] = useState(violation.telegram_sent === true);

  const handleSend = async () => {
    setLoading(true);
    try {
      const res = await fetch(`/api/alerts/telegram/${violation.id}`, {
        method: "POST",
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }

      setSent(true);
      onSuccess(`Alert sent to Telegram for plate ${violation.license_plate}!`);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Unknown error";
      onError(`Failed to send alert: ${msg}`);
    } finally {
      setLoading(false);
    }
  };

  if (sent) {
    return (
      <span className="inline-flex items-center gap-1.5 rounded-full bg-emerald-500/10 px-2.5 py-1 text-xs font-medium text-emerald-400 ring-1 ring-inset ring-emerald-500/25">
        <CheckCircle2 size={12} />
        Sent
      </span>
    );
  }

  return (
    <button
      onClick={handleSend}
      disabled={loading}
      title="Send Telegram Alert"
      className="inline-flex items-center gap-1.5 rounded-lg bg-sky-600/15 px-2.5 py-1.5 text-xs font-medium text-sky-400 ring-1 ring-inset ring-sky-500/25 transition hover:bg-sky-600/30 hover:text-sky-300 disabled:pointer-events-none disabled:opacity-50"
    >
      {loading ? (
        <RefreshCw size={12} className="animate-spin" />
      ) : (
        <Send size={12} />
      )}
      {loading ? "Sending…" : "Alert"}
    </button>
  );
}

/* ── Main Dashboard ────────────────────────────────────────── */

export default function Dashboard() {
  const { violations, loading, error, realtimeConnected, refetch } =
    useViolations();
  const [lightboxSrc, setLightboxSrc] = useState<string | null>(null);
  const { toasts, addToast, dismiss } = useToast();

  return (
    <div className="min-h-screen bg-gray-950">
      {/* ── Header ───────────────────────────────────────── */}
      <header className="sticky top-0 z-40 border-b border-gray-800 bg-gray-950/80 backdrop-blur-lg">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-3 sm:px-6">
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-to-br from-indigo-600 to-violet-600">
              <Car size={18} className="text-white" />
            </div>
            <div>
              <h1 className="text-lg font-bold leading-tight tracking-tight">
                Smart Parking Management
              </h1>
              <p className="text-[11px] text-gray-500">
                AI‑powered violation monitoring
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            {/* Real‑time status pill */}
            <span
              className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium ring-1 ring-inset ${
                realtimeConnected
                  ? "bg-emerald-500/10 text-emerald-400 ring-emerald-500/30"
                  : "bg-red-500/10 text-red-400 ring-red-500/30"
              }`}
            >
              <span
                className={`h-1.5 w-1.5 rounded-full ${
                  realtimeConnected
                    ? "bg-emerald-400 animate-live-pulse"
                    : "bg-red-400"
                }`}
              />
              {realtimeConnected ? "Live" : "Offline"}
              <Radio size={12} />
            </span>

            <button
              onClick={refetch}
              title="Refresh data"
              className="rounded-lg border border-gray-700 bg-gray-800/50 p-2 text-gray-400 transition hover:bg-gray-700 hover:text-white"
            >
              <RefreshCw size={15} />
            </button>
          </div>
        </div>
      </header>

      {/* ── Content ──────────────────────────────────────── */}
      <main className="mx-auto max-w-7xl space-y-6 px-4 py-6 sm:px-6">
        <StatsBar violations={violations} />

        {/* CCTV Monitor with ROI drawing */}
        <CCTVMonitor />

        {/* Error banner */}
        {error && (
          <div className="rounded-lg border border-red-800 bg-red-900/30 px-4 py-3 text-sm text-red-300">
            <strong>Error:</strong> {error}
          </div>
        )}

        {/* Table card */}
        <div className="overflow-hidden rounded-xl border border-gray-800 bg-gray-900/70 backdrop-blur">
          <div className="flex items-center justify-between border-b border-gray-800 px-5 py-3">
            <h2 className="text-sm font-semibold text-gray-200">
              Violation Records
            </h2>
            <span className="text-xs text-gray-500">
              {violations.length} record{violations.length !== 1 && "s"}
            </span>
          </div>

          {loading ? (
            <div className="flex items-center justify-center py-20 text-gray-500">
              <RefreshCw size={20} className="mr-2 animate-spin" />
              Loading violations…
            </div>
          ) : violations.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-20 text-gray-500">
              <ShieldAlert size={36} className="mb-3 opacity-40" />
              <p className="text-sm">No violations recorded yet.</p>
              <p className="text-xs text-gray-600">
                Run the AI pipeline to detect parking violations.
              </p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-gray-800 bg-gray-900/50 text-left text-xs font-medium uppercase tracking-wider text-gray-500">
                    <th className="px-5 py-3">ID</th>
                    <th className="px-5 py-3">License Plate</th>
                    <th className="px-5 py-3">Evidence</th>
                    <th className="px-5 py-3">Detected At</th>
                    <th className="px-5 py-3">Duration</th>
                    <th className="px-5 py-3">Status</th>
                    <th className="px-5 py-3">Alert</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-800/60">
                  {violations.map((v, idx) => (
                    <tr
                      key={v.id}
                      className={`transition hover:bg-gray-800/40 ${
                        idx === 0 ? "animate-slide-in" : ""
                      }`}
                    >
                      {/* ID */}
                      <td className="whitespace-nowrap px-5 py-3 font-mono text-xs text-gray-500">
                        #{v.id}
                      </td>

                      {/* Plate */}
                      <td className="whitespace-nowrap px-5 py-3">
                        <span className="inline-flex items-center gap-1.5 rounded-md bg-indigo-500/10 px-2.5 py-1 font-mono text-sm font-semibold text-indigo-300 ring-1 ring-inset ring-indigo-500/25">
                          <Car size={13} />
                          {v.license_plate}
                        </span>
                      </td>

                      {/* Evidence thumbnail */}
                      <td className="px-5 py-3">
                        {v.evidence_image_path ? (
                          <button
                            onClick={() =>
                              setLightboxSrc(v.evidence_image_path)
                            }
                            className="group relative overflow-hidden rounded-lg border border-gray-700 transition hover:border-indigo-500"
                          >
                            <img
                              src={v.evidence_image_path}
                              alt={`Evidence #${v.id}`}
                              loading="lazy"
                              className="h-10 w-16 object-cover transition group-hover:scale-110"
                            />
                            <div className="absolute inset-0 flex items-center justify-center bg-black/40 opacity-0 transition group-hover:opacity-100">
                              <ImageIcon size={14} className="text-white" />
                            </div>
                          </button>
                        ) : (
                          <span className="text-xs text-gray-600">
                            No image
                          </span>
                        )}
                      </td>

                      {/* Timestamp */}
                      <td className="whitespace-nowrap px-5 py-3 text-xs text-gray-400">
                        <span className="inline-flex items-center gap-1">
                          <Clock size={12} className="opacity-60" />
                          {bestTimestamp(v)}
                        </span>
                      </td>

                      {/* Duration */}
                      <td className="whitespace-nowrap px-5 py-3 text-xs text-gray-400">
                        {v.duration_seconds}s
                      </td>

                      {/* Status */}
                      <td className="whitespace-nowrap px-5 py-3">
                        <span
                          className={`inline-block rounded-full px-2.5 py-0.5 text-xs font-medium ring-1 ring-inset ${statusBadge(
                            v.status ?? "Pending"
                          )}`}
                        >
                          {v.status ?? "Pending"}
                        </span>
                      </td>

                      {/* Alert action */}
                      <td className="whitespace-nowrap px-5 py-3">
                        <SendAlertButton
                          violation={v}
                          onSuccess={(msg) => addToast(msg, "success")}
                          onError={(msg) => addToast(msg, "error")}
                        />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </main>

      {/* ── Lightbox overlay ─────────────────────────────── */}
      {lightboxSrc && (
        <Lightbox
          src={lightboxSrc}
          onClose={() => setLightboxSrc(null)}
        />
      )}

      {/* ── Toast notifications ───────────────────────────── */}
      <ToastContainer toasts={toasts} dismiss={dismiss} />
    </div>
  );
}
