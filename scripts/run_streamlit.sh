#!/bin/bash
# Streamlit 웹 데모 실행 스크립트

echo "Streamlit 웹 데모 시작..."
echo ""

# 프로젝트 루트로 이동
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Streamlit 실행
streamlit run app/web_demo.py --server.port 8501

