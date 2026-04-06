# IDS Web App (Converted)

This repository was extended with a minimal web app: a FastAPI backend that loads the existing Keras model and a React (Vite) frontend to send samples for prediction.

Quick overview
- Backend: `backend/app.py` (FastAPI). Serves `/predict` and `/classes`.
- Frontend: `frontend/` (Vite + React). Simple UI to enter samples and view predictions + a pie chart.

How to run (development)

1. Create a Python environment and install backend deps

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
pip install -r requirements.txt
```

If you see TensorFlow install errors, check your Python version. TensorFlow typically supports Python 3.10–3.11. If you're on a newer version, create the venv with a compatible Python (e.g., `python3.11 -m venv .venv`).

2. Start backend (from repo root)

```bash
python -m uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
```

Make sure `models/best_ann_model.keras` exists and `data/KDDTrain+.txt` and `data/KDDTest+.txt` are present. The backend will fit the preprocessing pipeline on the raw data files at startup.

3. Start frontend

```bash
cd frontend
npm install
npm run dev
```

Optional: run both backend and frontend together

```bash
chmod +x run-dev.sh
./run-dev.sh
```

Open the URL printed by Vite (usually http://localhost:5173).

Notes and next steps
- The backend fits the preprocessor from the original dataset at startup so it can accept unseen inputs without a separate saved transformer. For production, save the fitted transformer and label encoder to disk.
- Frontend is a minimal scaffold. You can improve validation, add CSV upload, or better UI components (Material UI, Tailwind, etc.).
