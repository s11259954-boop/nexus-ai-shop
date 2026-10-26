#!/usr/bin/env bash

# ========================================================
# Launch Script for AI Recommendation & Analytics Platform
# ========================================================

set -e

# Detect virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
fi

MODE="${1:-app}"

case "$MODE" in
    app)
        echo "🚀 Starting Streamlit Web Application on http://localhost:8501..."
        streamlit run app/app.py
        ;;
    api)
        echo "⚡ Starting FastAPI Backend on http://localhost:8000 (Docs at /docs)..."
        uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
        ;;
    all)
        echo "🌟 Starting both FastAPI Backend and Streamlit Frontend..."
        uvicorn api.main:app --host 0.0.0.0 --port 8000 &
        API_PID=$!
        sleep 2
        streamlit run app/app.py &
        APP_PID=$!
        echo "✅ Both services running! Press Ctrl+C to terminate."
        trap "kill $API_PID $APP_PID 2>/dev/null" EXIT
        wait
        ;;
    seed)
        echo "🌱 Seeding database and re-exporting datasets..."
        python data/seed_data.py
        python src/preprocessing.py
        python src/feature_engineering.py
        python dashboard/export_for_powerbi.py
        echo "✅ Seeding complete!"
        ;;
    test)
        echo "🧪 Running PyTest suite..."
        pytest -v tests/
        ;;
    *)
        echo "Usage: ./run.sh [app|api|all|seed|test]"
        exit 1
        ;;
esac
