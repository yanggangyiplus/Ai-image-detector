#!/bin/bash
# FastAPI 서버 실행 스크립트

echo "FastAPI 서버 시작..."
echo ""

# 프로젝트 루트로 이동
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# FastAPI 실행
python app/api.py

# 또는 uvicorn 직접 사용:
# uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload

