#!/bin/bash
# Docker 이미지 빌드 및 실행 스크립트
# 프로젝트 루트에서 실행해야 합니다

set -e

# 프로젝트 루트로 이동
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║          🐳 AI Image Detector Docker 빌드 및 실행               ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "프로젝트 루트: $PROJECT_ROOT"
echo ""

# 옵션 선택
MODE=${1:-"all"}

case $MODE in
  api)
    echo "📦 FastAPI 이미지 빌드 중..."
    docker build -f deployment/docker/Dockerfile.api -t ai-image-detector-api .
    echo ""
    echo "🚀 FastAPI 컨테이너 실행 중..."
    docker run -d --name ai-image-detector-api -p 8000:8000 ai-image-detector-api
    echo "✅ FastAPI 서버 실행 완료!"
    echo "👉 http://localhost:8000/docs"
    ;;
  
  streamlit)
    echo "📦 Streamlit 이미지 빌드 중..."
    docker build -f deployment/docker/Dockerfile.streamlit -t ai-image-detector-streamlit .
    echo ""
    echo "🚀 Streamlit 컨테이너 실행 중..."
    docker run -d --name ai-image-detector-streamlit -p 8501:8501 ai-image-detector-streamlit
    echo "✅ Streamlit 데모 실행 완료!"
    echo "👉 http://localhost:8501"
    ;;
  
  compose)
    echo "📦 Docker Compose로 빌드 및 실행 중..."
    cd deployment/docker
    docker-compose build
    docker-compose up -d
    cd "$PROJECT_ROOT"
    echo ""
    echo "✅ 모든 서비스 실행 완료!"
    echo "👉 FastAPI: http://localhost:8000/docs"
    echo "👉 Streamlit: http://localhost:8501"
    ;;
  
  all|*)
    echo "📦 모든 이미지 빌드 중..."
    docker build -f deployment/docker/Dockerfile.api -t ai-image-detector-api .
    docker build -f deployment/docker/Dockerfile.streamlit -t ai-image-detector-streamlit .
    echo ""
    echo "✅ 빌드 완료!"
    echo ""
    echo "실행 방법:"
    echo "  cd deployment/docker"
    echo "  ./build_and_run.sh api        # FastAPI만 실행"
    echo "  ./build_and_run.sh streamlit  # Streamlit만 실행"
    echo "  ./build_and_run.sh compose    # Docker Compose로 모두 실행"
    ;;
esac

echo ""
echo "📋 유용한 명령어:"
echo "  docker ps                        # 실행 중인 컨테이너 확인"
echo "  docker logs ai-image-detector-api    # API 로그 확인"
echo "  docker logs ai-image-detector-streamlit  # Streamlit 로그 확인"
echo "  docker-compose logs -f           # 모든 로그 확인"
echo "  docker-compose down             # 모든 서비스 중지"
