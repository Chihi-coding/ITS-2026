"""
Interactive ROI polygon drawing tool.

Open a video, click to define polygon vertices on the first frame,
then press 'c' to save coordinates to ai_module/config/roi.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT_DIR / "config" / "roi.json"
WINDOW_NAME = "Draw ROI - click points, press 'c' to save, 'r' to reset, 'q' to quit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw a Region of Interest polygon on the first frame of a video."
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Path to a video file or webcam index (e.g. 0).",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=f"Output JSON path (default: {DEFAULT_OUTPUT}).",
    )
    return parser.parse_args()


def resolve_source(source: str) -> str | int:
    if source.isdigit():
        return int(source)
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Video source not found: {source}")
    return str(path.resolve())


def on_mouse(event: int, x: int, y: int, _flags: int, userdata: dict) -> None:
    if event == cv2.EVENT_LBUTTONDOWN:
        userdata["points"].append([x, y])
        print(f"Point {len(userdata['points'])}: ({x}, {y})")


def draw_polygon(frame, points: list[list[int]]) -> None:
    if not points:
        return

    for index, point in enumerate(points):
        cv2.circle(frame, tuple(point), 5, (0, 255, 255), -1)
        if index > 0:
            cv2.line(frame, tuple(points[index - 1]), tuple(point), (0, 255, 0), 2)

    if len(points) >= 3:
        cv2.line(frame, tuple(points[-1]), tuple(points[0]), (0, 255, 0), 2)
        overlay = frame.copy()
        polygon = np.array(points, dtype=np.int32)
        cv2.fillPoly(overlay, [polygon], color=(0, 255, 0))
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)


def save_polygon(output_path: Path, points: list[list[int]]) -> None:
    if len(points) < 3:
        raise ValueError("At least 3 points are required to define a polygon.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(points, handle, indent=2)

    print(f"Saved {len(points)} ROI points to {output_path}")


def main() -> None:
    args = parse_args()
    source = resolve_source(args.source)
    output_path = Path(args.output)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Unable to open video source: {args.source}", file=sys.stderr)
        sys.exit(1)

    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        print("Failed to read the first frame from the video source.", file=sys.stderr)
        sys.exit(1)

    state = {"points": []}
    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse, state)

    print("Instructions:")
    print("  - Left-click to add polygon vertices")
    print("  - Press 'r' to reset points")
    print("  - Press 'c' to save polygon and exit")
    print("  - Press 'q' or ESC to quit without saving")

    while True:
        display = frame.copy()
        draw_polygon(display, state["points"])

        cv2.putText(
            display,
            f"Points: {len(state['points'])} | 'c'=save 'r'=reset 'q'=quit",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(WINDOW_NAME, display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            print("Exited without saving.")
            break
        if key == ord("r"):
            state["points"].clear()
            print("Reset polygon points.")
        if key == ord("c"):
            try:
                save_polygon(output_path, state["points"])
            except ValueError as exc:
                print(exc, file=sys.stderr)
                continue
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
