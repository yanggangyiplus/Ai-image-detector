# Docker 패키징 가이드

이 디렉토리에는 Docker 관련 파일들이 정리되어 있습니다.

## 📁 파일 구조

```
deployment/docker/
├── Dockerfile              # 기본 Dockerfile (FastAPI)
├── Dockerfile.api          # FastAPI 전용
├── Dockerfile.streamlit    # Streamlit 전용
├── docker-compose.yml      # Docker Compose 설정
├── .dockerignore           # 빌드 제외 파일 목록
├── build_and_run.sh        # 빌드 및 실행 스크립트
└── README.md               # 이 파일
```

## 🚀 빠른 시작

### 1. Docker 설치

#### macOS
```bash
# Homebrew 사용
brew install --cask docker

# 또는 Docker Desktop 직접 다운로드
# https://www.docker.com/products/docker-desktop
```

설치 후 Docker Desktop을 실행하고 완전히 시작될 때까지 기다리세요.

#### 설치 확인
```bash
docker --version
docker-compose --version
```

### 2. Docker 이미지 빌드

#### 프로젝트 루트에서 실행:

```bash
# FastAPI만 빌드
docker build -f deployment/docker/Dockerfile.api -t ai-image-detector-api .

# Streamlit만 빌드
docker build -f deployment/docker/Dockerfile.streamlit -t ai-image-detector-streamlit .

# 또는 스크립트 사용
cd deployment/docker
./build_and_run.sh all
```

### 3. 컨테이너 실행

#### 개별 실행:

```bash
# FastAPI
docker run -d --name ai-image-detector-api -p 8000:8000 ai-image-detector-api

# Streamlit
docker run -d --name ai-image-detector-streamlit -p 8501:8501 ai-image-detector-streamlit
```

#### Docker Compose 사용 (권장):

```bash
cd deployment/docker
docker-compose up -d
```

### 4. 접속

- **FastAPI**: http://localhost:8000/docs
- **Streamlit**: http://localhost:8501

## 📋 사용 방법

### Docker Compose 사용법

```bash
# 서비스 시작 (백그라운드)
cd deployment/docker
docker-compose up -d

# 서비스 시작 (포그라운드, 로그 확인)
docker-compose up

# 서비스 중지
docker-compose stop

# 서비스 중지 및 제거
docker-compose down

# 로그 확인
docker-compose logs -f
docker-compose logs -f api
docker-compose logs -f streamlit

# 서비스 재시작
docker-compose restart
```

### 스크립트 사용법

```bash
cd deployment/docker

# 모든 이미지 빌드
./build_and_run.sh all

# FastAPI만 빌드 및 실행
./build_and_run.sh api

# Streamlit만 빌드 및 실행
./build_and_run.sh streamlit

# Docker Compose로 모두 실행
./build_and_run.sh compose
```

## 🔧 고급 사용법

### 볼륨 마운트

체크포인트를 외부에서 업데이트하려면:

```bash
docker run -d \
  --name ai-image-detector-api \
  -p 8000:8000 \
  -v $(pwd)/experiments/checkpoints:/app/experiments/checkpoints:ro \
  ai-image-detector-api
```

### 환경 변수 설정

```bash
docker run -d \
  --name ai-image-detector-api \
  -p 8000:8000 \
  -e PYTHONUNBUFFERED=1 \
  -e LOG_LEVEL=INFO \
  ai-image-detector-api
```

## 🐛 문제 해결

### 포트가 이미 사용 중일 때

```bash
# 다른 포트 사용
docker run -d -p 8001:8000 ai-image-detector-api
```

### 컨테이너 로그 확인

```bash
docker logs ai-image-detector-api
docker logs -f ai-image-detector-api  # 실시간 로그
```

### 컨테이너 내부 접속

```bash
docker exec -it ai-image-detector-api bash
```

### 이미지/컨테이너 정리

```bash
# 중지된 컨테이너 제거
docker container prune

# 사용하지 않는 이미지 제거
docker image prune

# 모든 것 정리 (주의!)
docker system prune -a
```

## 📊 이미지 최적화

현재 Dockerfile은 다음 최적화를 포함합니다:

- ✅ Multi-stage 빌드
- ✅ .dockerignore로 불필요한 파일 제외
- ✅ 레이어 캐싱 최적화
- ✅ 최소한의 시스템 패키지만 설치

## 🔐 프로덕션 배포

프로덕션 환경에서는 다음을 고려하세요:

1. **보안**: 비root 사용자 설정
2. **리소스 제한**: CPU/메모리 제한 설정
3. **로깅**: 로그 드라이버 설정
4. **모니터링**: 헬스 체크 설정 (이미 포함됨)

## 📝 체크리스트

- [x] Dockerfile 작성
- [x] .dockerignore 작성
- [x] docker-compose.yml 작성
- [x] 빌드 스크립트 작성
- [x] 경로 정리 완료

## 🔗 참고 자료

- [Docker 공식 문서](https://docs.docker.com/)
- [Docker Compose 문서](https://docs.docker.com/compose/)
- [Best Practices](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/)

