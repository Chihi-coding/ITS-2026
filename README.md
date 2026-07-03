# Smart Parking System (ITS-2026)

Intelligent Transportation System (ITS) for detecting illegal parking violations using YOLOv8, FastAPI, React (Vite), and Supabase.

## Project Structure

```
smart_parking_system/
├── ai_module/          # Computer vision pipeline (YOLOv8, OpenCV, EasyOCR)
├── backend/            # FastAPI REST API
├── frontend/           # React + Vite Dashboard with TailwindCSS
├── requirements.txt    # Python dependencies for Backend and AI
└── README.md
```

## Prerequisites

- **Python 3.11+**
- **Node.js (v18+) & npm**
- **Supabase Project** (Database & Storage)
- **Telegram Bot Token & Chat ID** (optional, for alerts)

## Setup & Installation

### 1. Supabase Database Setup
In your Supabase SQL Editor, run the contents of the `backend/setup_supabase.sql` file to create the necessary tables, storage buckets, and security policies.

### 2. Backend & AI Module (Python)

1. From the project root, create and activate a virtual environment:
   ```bash
   python -m venv .venv
   
   # Windows:
   .venv\Scripts\activate
   # Mac/Linux:
   source .venv/bin/activate
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure Backend Environment Variables:
   Create a `.env` file in the `backend/` folder (you can copy `backend/.env.example`) and fill in your keys:
   ```env
   # Example backend/.env
   DATABASE_URL=postgresql://postgres:[YOUR-PASSWORD]@db.[YOUR-PROJECT].supabase.co:5432/postgres
   SUPABASE_URL=https://[YOUR-PROJECT].supabase.co
   SUPABASE_KEY=[YOUR-SUPABASE-SERVICE-ROLE-KEY]
   
   # Optional Telegram integration
   TELEGRAM_BOT_TOKEN=your_token
   TELEGRAM_CHAT_ID=your_chat_id
   ```

### 3. Frontend Dashboard (React/Vite)

1. Navigate to the `frontend/` directory and install NPM packages:
   ```bash
   cd frontend
   npm install
   ```

2. Configure Frontend Environment Variables:
   Copy `frontend/.env.example` to `frontend/.env` and update it with your Supabase keys:
   ```env
   # Example frontend/.env
   VITE_SUPABASE_URL=https://[YOUR-PROJECT].supabase.co
   VITE_SUPABASE_ANON_KEY=[YOUR-SUPABASE-ANON-KEY]
   ```

---

## How to Run the Project

You will need to open **three separate terminals** to run all components of the system simultaneously.

### Terminal 1: Start the Backend API
Make sure your Python virtual environment is activated.
```bash
cd backend
uvicorn app.main:app --reload --port 8000
```
*API documentation will be available at: http://127.0.0.1:8000/docs*

### Terminal 2: Start the Frontend App
```bash
cd frontend
npm run dev
```
*The web dashboard will be available at: http://localhost:5173*

### Terminal 3: Start the AI Module
Make sure your Python virtual environment is activated and you are at the project root.
```bash
python -m ai_module.main --source 0 --backend-url http://127.0.0.1:8000
```
*Options:*
- `--source`: Camera index (e.g., `0` for webcam) or path to a video file (e.g., `test_data/video.mp4`).
- `--roi-config`: Path to the ROI polygon JSON file.

## Violation Flow Summary

1. The **AI module** detects vehicles using YOLOv8.
2. It monitors vehicles inside a defined Region of Interest (ROI).
3. If a vehicle's dwell time exceeds the allowed limit (e.g., 10 seconds), OCR extracts the license plate text.
4. The cropped image and metadata are POSTed to the **Backend API** (`/api/violations`).
5. The Backend stores the data in **Supabase Postgres**, uploads the image to **Supabase Storage**, and sends a Telegram alert.
6. The **Frontend dashboard** listens to Supabase for real-time updates and displays the new violation instantly.
