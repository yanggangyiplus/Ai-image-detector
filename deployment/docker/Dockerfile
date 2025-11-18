# AI Image Detector - Multi-stage Dockerfile
# 포트폴리오용 최적화된 Docker 이미지

# Stage 1: Base image with Python
FROM python:3.11-slim as base

# 환경 변수 설정
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 작업 디렉토리 설정
WORKDIR /app

# Stage 2: Dependencies
FROM base as dependencies

# 시스템 패키지 설치 (이미지 처리 및 기타 유틸리티)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성 파일 복사
COPY requirements.txt .

# PyTorch CPU 버전 설치 (GPU는 별도 이미지 필요)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 나머지 의존성 설치
RUN pip install --no-cache-dir -r requirements.txt

# Stage 3: Production
FROM dependencies as production

# 애플리케이션 코드 복사
COPY src/ ./src/
COPY app/ ./app/
COPY configs/ ./configs/
COPY experiments/checkpoints/ ./experiments/checkpoints/
COPY data/class_map.json ./data/
COPY data/metadata.json ./data/

# 실행 스크립트 복사
COPY run_api.sh run_streamlit.sh ./
RUN chmod +x run_api.sh run_streamlit.sh

# 포트 노출 (FastAPI: 8000, Streamlit: 8501)
EXPOSE 8000 8501

# 기본 명령어 (FastAPI 서버)
CMD ["python", "app/api.py"]

# 사용 예시:
# FastAPI 실행: docker run -p 8000:8000 ai-image-detector
# Streamlit 실행: docker run -p 8501:8501 ai-image-detector streamlit run app/web_demo.py --server.port 8501

