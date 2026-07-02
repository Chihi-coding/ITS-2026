"""
Interactive ROI Setup Tool.

Extracts the first frame from a video file, lets you click 4 points to
define the parking-zone polygon, and saves the coordinates to
ai_module/config/roi.json.

Usage:
    python -m ai_module.setup_roi              # defaults to test.mp4
    python -m ai_module.setup_roi --source path/to/video.mp4

Controls:
    Left-click   Add a polygon vertex (up to 4)
    'r'          Reset all points
    'c'          Confirm and save polygon to roi.json
    'q' / ESC    Quit without saving
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT_DIR.parent
CONFIG_DIR = ROOT_DIR / "config"
DEFAULT_OUTPUT = CONFIG_DIR / "roi.json"
DEFAULT_VIDEO = PROJECT_ROOT / "test.mp4"
WINDOW_NAME = "ROI Setup - click 4 corners, 'c'=save, 'r'=reset, 'q'=quit"
MAX_POINTS = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw a parking-zone ROI on the first frame of a video."
    )
    parser.add_argument(
        "--source",
        default=str(DEFAULT_VIDEO),
        help=f"Path to a video file (default: {DEFAULT_VIDEO}).",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=f"Output JSON path (default: {DEFAULT_OUTPUT}).",
    )
    return parser.parse_args()


def on_mouse(event: int, x: int, y: int, _flags: int, userdata: dict) -> None:
    """Record left-clicks as polygon vertices (max MAX_POINTS)."""
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(userdata["points"]) >= MAX_POINTS:
            print(f"[INFO] Already have {MAX_POINTS} points. Press 'r' to reset or 'c' to save.")
            return
        userdata["points"].append([x, y])
        print(f"  Point {len(userdata['points'])}: ({x}, {y})")


def draw_polygon(frame: np.ndarray, points: list[list[int]]) -> None:
    """Draw the polygon-in-progress on the frame (in-place)."""
    if not points:
        return

    # Draw each vertex as a circle
    for idx, pt in enumerate(points):
        cv2.circle(frame, tuple(pt), 6, (0, 255, 255), -1)
        cv2.putText(
            frame,
            str(idx + 1),
            (pt[0] + 8, pt[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        # Connect consecutive vertices
        if idx > 0:
            cv2.line(frame, tuple(points[idx - 1]), tuple(pt), (0, 255, 0), 2)

    # Close the polygon and fill it with a transparent overlay
    if len(points) >= 3:
        cv2.line(frame, tuple(points[-1]), tuple(points[0]), (0, 255, 0), 2)
        overlay = frame.copy()
        polygon_arr = np.array(points, dtype=np.int32)
        cv2.fillPoly(overlay, [polygon_arr], color=(0, 255, 0))
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)


def save_polygon(output_path: Path, points: list[list[int]]) -> None:
    """Validate and save the polygon to JSON."""
    if len(points) < 3:
        raise ValueError("Need at least 3 points to define a polygon.")
    if len(points) != MAX_POINTS:
        raise ValueError(f"Expected exactly {MAX_POINTS} points, got {len(points)}.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(points, f, indent=2)

    print(f"\n{'=' * 50}")
    print(f"  ROI saved to {output_path}")
    print(f"  Coordinates: {points}")
    print(f"{'=' * 50}\n")


def main() -> None:
    args = parse_args()
    source_path = Path(args.source)
    output_path = Path(args.output)

    if not source_path.exists():
        print(f"[ERROR] Video file not found: {source_path}", file=sys.stderr)
        sys.exit(1)

    # Extract the first frame
    cap = cv2.VideoCapture(str(source_path.resolve()))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {source_path}", file=sys.stderr)
        sys.exit(1)

    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        print("[ERROR] Failed to read the first frame.", file=sys.stderr)
        sys.exit(1)

    h, w = frame.shape[:2]
    print(f"\n[INFO] Video resolution: {w}x{h}")

    # Auto-scale to fit on screen (max 800px height, preserve aspect ratio)
    MAX_WIN_HEIGHT = 800
    scale = MAX_WIN_HEIGHT / h
    win_w, win_h = int(w * scale), MAX_WIN_HEIGHT
    print(f"[INFO] Display scaled to {win_w}x{win_h} (scale={scale:.3f})")

    print(f"[INFO] Click {MAX_POINTS} corners of the parking zone on the image.")
    print("[INFO] Press 'c' to confirm, 'r' to reset, 'q'/ESC to quit.\n")

    # Store scale in state so on_mouse can map display coords → original coords
    state: dict = {"points": [], "scale": scale}
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, win_w, win_h)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse, state)

    while True:
        display = frame.copy()
        draw_polygon(display, state["points"])

        # HUD text
        n = len(state["points"])
        status = f"Points: {n}/{MAX_POINTS}"
        if n == MAX_POINTS:
            status += "  |  Press 'c' to SAVE"
        cv2.putText(
            display, status, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
        )

        cv2.imshow(WINDOW_NAME, display)

        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            print("[INFO] Exited without saving.")
            break
        elif key == ord("r"):
            state["points"].clear()
            print("[INFO] Points reset.")
        elif key == ord("c"):
            try:
                save_polygon(output_path, state["points"])
            except ValueError as exc:
                print(f"[WARN] {exc}")
                continue
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
