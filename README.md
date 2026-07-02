# Smart Parking System

Intelligent Transportation System (ITS) for detecting illegal parking violations with YOLOv8, FastAPI, PostgreSQL/SQLite, and Telegram alerts.

## Project Structure

```
smart_parking_system/
├── ai_module/          # Computer vision pipeline
├── backend/            # FastAPI REST API
├── frontend/           # Bootstrap dashboard
├── requirements.txt    # All Python deps (backend + AI)
└── docker-compose.yml
```

## Prerequisites

- Python 3.11+
- Optional: Docker for PostgreSQL
- Telegram Bot token and chat ID for alerts

## Install (backend + AI, one command)

From the project root:

```bash
cd D:\code\ITS-illegal-parking
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Copy backend environment config:

```bash
copy backend\.env.example backend\.env
```

Edit `backend\.env`:

```env
DATABASE_URL=sqlite:///./parking.db
TELEGRAM_BOT_TOKEN=your_token
TELEGRAM_CHAT_ID=your_chat_id
```

## Backend

```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

API docs: http://127.0.0.1:8000/docs

## Frontend

Open `frontend/index.html` with Live Server or any static file server. The dashboard polls `GET /api/violations` every 5 seconds.

Ensure CORS and API base URL in `frontend/js/app.js` match your backend host.

## AI Module

From the project root (with the same venv activated):

```bash
python -m ai_module.main --source 0 --backend-url http://127.0.0.1:8000
```

Options:

- `--source`: camera index or video file path
- `--roi-config`: path to ROI polygon JSON
- `--vehicle-model`: YOLOv8 vehicle weights
- `--plate-model`: optional dedicated plate detector weights

## ROI Configuration

Edit `ai_module/config/roi_zone.json`:

```json
{
  "max_dwell_seconds": 10,
  "polygon": [[320, 180], [960, 180], [1020, 520], [260, 520]]
}
```

## Violation Flow

1. AI module detects vehicles with YOLOv8.
2. Tracker monitors bottom-center point inside ROI using `cv2.pointPolygonTest`.
3. If dwell time exceeds 10 seconds, OCR extracts plate text.
4. Cropped image and metadata are POSTed to `/api/violations`.
5. Backend stores data, saves image, and sends Telegram alert.
6. Frontend dashboard displays violations in near real time.

## Docker (PostgreSQL + Backend)

```bash
docker compose up -d
```

Use PostgreSQL URL in `backend/.env` when running with Docker.
