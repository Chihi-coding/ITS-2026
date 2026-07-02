import { useState, useRef, useEffect, useCallback } from "react";
import { Video, Crosshair, Save, RotateCcw, Check, AlertTriangle } from "lucide-react";

/* ── Types ──────────────────────────────────────────────────── */

type Point = [number, number]; // native video pixel coords

type ToastState = { message: string; type: "success" | "error" } | null;

/* ── Constants ──────────────────────────────────────────────── */

const MAX_POINTS = 4;
const STREAM_URL = "/api/video_feed";
const ROI_API = "/api/roi";
const VIDEO_INFO_API = "/api/video_info";

/* ─────────────────────────────────────────────────────────────
   Component
   ───────────────────────────────────────────────────────────── */

export default function CCTVMonitor() {
  const [points, setPoints] = useState<Point[]>([]);
  const [saving, setSaving] = useState(false);
  const [toast, setToast] = useState<ToastState>(null);
  const [drawMode, setDrawMode] = useState(false);

  // We only need the *natural* (video) resolution — the SVG is
  // laid out in a 1×1 viewBox with preserveAspectRatio="none"
  // so display ↔ native mapping is a simple ratio of the container rect.
  const [naturalSize, setNaturalSize] = useState({ w: 0, h: 0 });
  // Dynamic aspect ratio from cropped video dimensions
  const [videoAspect, setVideoAspect] = useState("16/9");

  const imgRef = useRef<HTMLImageElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  /* ── Load saved ROI on mount ──────────────────────────────── */
  useEffect(() => {
    fetch(ROI_API)
      .then((r) => r.json())
      .then((data) => {
        if (Array.isArray(data.points) && data.points.length > 0)
          setPoints(data.points as Point[]);
      })
      .catch(() => {});
  }, []);

  /* ── Fetch cropped video dimensions for dynamic aspect ratio ── */
  useEffect(() => {
    fetch(VIDEO_INFO_API)
      .then((r) => r.json())
      .then((data) => {
        if (data.width && data.height) {
          setVideoAspect(`${data.width}/${data.height}`);
        }
      })
      .catch(() => {});
  }, []);

  /* ── Detect natural video resolution once stream starts ───── */
  const captureNaturalSize = useCallback(() => {
    const img = imgRef.current;
    if (img && img.naturalWidth > 0 && naturalSize.w === 0) {
      setNaturalSize({ w: img.naturalWidth, h: img.naturalHeight });
    }
  }, [naturalSize.w]);

  useEffect(() => {
    const img = imgRef.current;
    if (!img) return;
    img.addEventListener("load", captureNaturalSize);
    // Poll until the MJPEG stream delivers the first frame
    const id = setInterval(captureNaturalSize, 400);
    return () => {
      img.removeEventListener("load", captureNaturalSize);
      clearInterval(id);
    };
  }, [captureNaturalSize]);

  /* ─────────────────────────────────────────────────────────────
     Coordinate helpers

     The container div has `position:relative` and the img fills it
     100%×100% (object-fit:cover).  The SVG overlay also fills 100%
     of the container.  We map clicks → native video pixels by:

       native_x = click_offset_x / container_width  * natural_w
       native_y = click_offset_y / container_height * natural_h

     And the reverse for drawing:

       display_x = (native_x / natural_w)  — stored as a fraction [0,1]
                   then used with SVG viewBox="0 0 1 1" and
                   preserveAspectRatio="none".

     Using fractions lets the SVG scale perfectly at any window size
     without any React state for the rendered image dimensions.
   ───────────────────────────────────────────────────────────── */

  /** Native-pixel point → SVG [0,1] fraction */
  const toFrac = (pt: Point): [number, number] => {
    if (naturalSize.w === 0) return [0, 0];
    return [pt[0] / naturalSize.w, pt[1] / naturalSize.h];
  };

  /** SVG click → native-pixel point */
  const clickToNative = (e: React.MouseEvent<SVGSVGElement>): Point => {
    const rect = containerRef.current!.getBoundingClientRect();
    const fx = (e.clientX - rect.left) / rect.width;
    const fy = (e.clientY - rect.top) / rect.height;
    return [
      Math.round(fx * (naturalSize.w || 1)),
      Math.round(fy * (naturalSize.h || 1)),
    ];
  };

  /* ── SVG click handler ────────────────────────────────────── */
  const handleSvgClick = (e: React.MouseEvent<SVGSVGElement>) => {
    if (!drawMode || points.length >= MAX_POINTS) return;
    setPoints((prev) => [...prev, clickToNative(e)]);
  };

  /* ── Save ROI ─────────────────────────────────────────────── */
  const handleSave = async () => {
    if (points.length !== MAX_POINTS) return;
    setSaving(true);
    try {
      const res = await fetch(ROI_API, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ points }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || "Save failed");
      }
      showToast("ROI saved successfully!", "success");
    } catch (err: unknown) {
      showToast(err instanceof Error ? err.message : "Unknown error", "error");
    } finally {
      setSaving(false);
    }
  };

  const showToast = (message: string, type: "success" | "error") => {
    setToast({ message, type });
    setTimeout(() => setToast(null), 3000);
  };

  /* ── Derived SVG values ───────────────────────────────────── */
  const isClosed = points.length === MAX_POINTS;

  // Points as "x,y" fractions for SVG (viewBox 0 0 1 1)
  const polyPoints = points.map((p) => toFrac(p).join(",")).join(" ");

  /* ── Render ───────────────────────────────────────────────── */
  return (
    <div className="overflow-hidden rounded-xl border border-gray-800 bg-gray-900/70 backdrop-blur">

      {/* ── Header ──────────────────────────────────────── */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-gray-800 px-5 py-3">
        <div className="flex items-center gap-2.5">
          <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-gradient-to-br from-cyan-600 to-blue-600">
            <Video size={15} className="text-white" />
          </div>
          <div>
            <h2 className="text-sm font-semibold text-gray-200">CCTV Monitor</h2>
            <p className="text-[11px] text-gray-500">
              {drawMode
                ? `Click ${MAX_POINTS - points.length} more point${MAX_POINTS - points.length !== 1 ? "s" : ""} to define the zone`
                : "Live feed — press Draw ROI to start"}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {/* Draw-mode toggle */}
          <button
            onClick={() => {
              setDrawMode((d) => !d);
              if (!drawMode) setPoints([]);
            }}
            className={`inline-flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium transition ${
              drawMode
                ? "bg-cyan-600/20 text-cyan-400 ring-1 ring-inset ring-cyan-500/40 hover:bg-cyan-600/30"
                : "bg-gray-800 text-gray-300 ring-1 ring-inset ring-gray-700 hover:bg-gray-700 hover:text-white"
            }`}
          >
            <Crosshair size={13} />
            {drawMode ? "Drawing…" : "Draw ROI"}
          </button>

          {/* Reset */}
          <button
            onClick={() => setPoints([])}
            disabled={points.length === 0}
            className="inline-flex items-center gap-1.5 rounded-lg bg-gray-800 px-3 py-1.5 text-xs font-medium text-gray-300 ring-1 ring-inset ring-gray-700 transition hover:bg-gray-700 hover:text-white disabled:pointer-events-none disabled:opacity-40"
          >
            <RotateCcw size={13} />
            Reset
          </button>

          {/* Save */}
          <button
            onClick={handleSave}
            disabled={!isClosed || saving}
            className="inline-flex items-center gap-1.5 rounded-lg bg-gradient-to-r from-emerald-600 to-green-600 px-3.5 py-1.5 text-xs font-semibold text-white shadow-lg shadow-emerald-900/30 transition hover:from-emerald-500 hover:to-green-500 disabled:pointer-events-none disabled:opacity-40"
          >
            <Save size={13} />
            {saving ? "Saving…" : "Save ROI"}
          </button>
        </div>
      </div>

      {/* ── Video + SVG overlay ──────────────────────────── */}
      {/*
        Key layout decisions:
        - aspect-ratio: 16/9 on the container → fixed proportional box
        - The <img> fills 100%×100% with object-fit:cover → no letterbox bars
        - The <svg> is absolutely positioned to fill the same box exactly
        - SVG viewBox="0 0 1 1" with preserveAspectRatio="none" →
          coordinates are pure fractions; no React state for px dimensions
      */}
      <div
        ref={containerRef}
        style={{ aspectRatio: videoAspect }}
        className={`relative w-full overflow-hidden bg-black ${drawMode ? "cursor-crosshair" : ""}`}
      >
        {/* MJPEG stream — fills box fully, no padding, no letterbox */}
        <img
          ref={imgRef}
          src={STREAM_URL}
          alt="CCTV Live Feed"
          style={{ objectFit: "contain" }}
          className="absolute inset-0 h-full w-full"
          onLoad={captureNaturalSize}
        />

        {/* SVG overlay — always covers the img exactly */}
        <svg
          className="absolute inset-0 h-full w-full"
          viewBox="0 0 1 1"
          preserveAspectRatio="none"
          onClick={handleSvgClick}
        >
          {/* Filled / dashed polygon */}
          {points.length >= 3 && (
            <polygon
              points={polyPoints}
              fill="rgba(34, 197, 94, 0.18)"
              stroke="#22c55e"
              strokeWidth="0.003"
              strokeDasharray={isClosed ? undefined : "0.012 0.006"}
            />
          )}

          {/* 2-point line */}
          {points.length === 2 && (() => {
            const [ax, ay] = toFrac(points[0]);
            const [bx, by] = toFrac(points[1]);
            return (
              <line
                x1={ax} y1={ay} x2={bx} y2={by}
                stroke="#22c55e"
                strokeWidth="0.003"
                strokeDasharray="0.012 0.006"
              />
            );
          })()}

          {/* Vertex circles + labels — all in [0,1] space */}
          {points.map((pt, i) => {
            const [fx, fy] = toFrac(pt);
            return (
              <g key={i}>
                <circle cx={fx} cy={fy} r={0.018} fill="rgba(34,197,94,0.25)" />
                <circle cx={fx} cy={fy} r={0.009} fill="#22c55e" stroke="#fff" strokeWidth="0.003" />
                <text
                  x={fx + 0.022}
                  y={fy - 0.016}
                  fill="#fff"
                  fontSize="0.04"
                  fontWeight="600"
                  fontFamily="Inter, system-ui, sans-serif"
                  style={{ filter: "drop-shadow(0 1px 3px rgba(0,0,0,0.9))" }}
                >
                  {i + 1}
                </text>
              </g>
            );
          })}
        </svg>

        {/* REC badge */}
        <div className="absolute top-3 left-3 flex items-center gap-1.5 rounded-full bg-black/60 px-2.5 py-1 text-[11px] font-medium text-gray-200 backdrop-blur-sm ring-1 ring-inset ring-white/10">
          <span className="h-1.5 w-1.5 rounded-full bg-red-500 animate-live-pulse" />
          REC
        </div>

        {/* Point counter */}
        {drawMode && (
          <div className="absolute bottom-3 left-3 rounded-lg bg-black/60 px-3 py-1.5 text-xs font-medium text-gray-200 backdrop-blur-sm ring-1 ring-inset ring-white/10">
            Points: {points.length}/{MAX_POINTS}
            {isClosed && <span className="ml-2 text-emerald-400">✓ Ready to save</span>}
          </div>
        )}
      </div>

      {/* ── Toast ────────────────────────────────────────── */}
      {toast && (
        <div
          className={`flex items-center gap-2 px-5 py-2.5 text-sm font-medium animate-slide-in ${
            toast.type === "success"
              ? "bg-emerald-500/10 text-emerald-400"
              : "bg-red-500/10 text-red-400"
          }`}
        >
          {toast.type === "success" ? <Check size={15} /> : <AlertTriangle size={15} />}
          {toast.message}
        </div>
      )}
    </div>
  );
}
