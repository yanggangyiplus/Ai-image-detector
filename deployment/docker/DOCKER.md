# Docker 패키징 가이드

이 프로젝트는 Docker를 사용하여 컨테이너화되어 있습니다.

## 📋 파일 구조

```
.
├── Dockerfile              # 기본 Dockerfile (FastAPI)
├── Dockerfile.api          # FastAPI 전용
├── Dockerfile.streamlit    # Streamlit 전용
├── docker-compose.yml      # Docker Compose 설정
├── .dockerignore           # Docker 빌드 제외 파일
└── requirements.txt        # Python 의존성
```

## 🚀 빠른 시작

### 1. Docker 이미지 빌드

#### FastAPI 서버만 빌드
```bash
docker build -f Dockerfile.api -t ai-image-detector-api .
```

#### Streamlit 데모만 빌드
```bash
docker build -f Dockerfile.streamlit -t ai-image-detector-streamlit .
```

#### 모든 서비스 빌드 (Docker Compose 사용)
```bash
docker-compose build
```

### 2. 컨테이너 실행

#### FastAPI 서버 실행
```bash
docker run -d \
  --name ai-image-detector-api \
  -p 8000:8000 \
  ai-image-detector-api
```

#### Streamlit 데모 실행
```bash
docker run -d \
  --name ai-image-detector-streamlit \
  -p 8501:8501 \
  ai-image-detector-streamlit
```

#### Docker Compose로 모두 실행
```bash
docker-compose up -d
```

### 3. 접속

- **FastAPI**: http://localhost:8000
  - API 문서: http://localhost:8000/docs
  - ReDoc: http://localhost:8000/redoc
  - 헬스 체크: http://localhost:8000/health

- **Streamlit**: http://localhost:8501

## 📦 Docker Compose 사용법

### 서비스 시작
```bash
# 백그라운드 실행
docker-compose up -d

# 포그라운드 실행 (로그 확인)
docker-compose up
```

### 서비스 중지
```bash
# 중지
docker-compose stop

# 중지 및 제거
docker-compose down
```

### 로그 확인
```bash
# 모든 서비스 로그
docker-compose logs -f

# 특정 서비스 로그
docker-compose logs -f api
docker-compose logs -f streamlit
```

### 서비스 재시작
```bash
docker-compose restart
```

## 🔧 고급 사용법

### 볼륨 마운트 (체크포인트 업데이트)

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

### GPU 사용 (NVIDIA Docker)

GPU를 사용하려면 `nvidia-docker`가 필요합니다:

```bash
# Dockerfile 수정 필요: CPU 버전 대신 GPU 버전 설치
docker run --gpus all -d \
  --name ai-image-detector-api \
  -p 8000:8000 \
  ai-image-detector-api
```

## 🐛 문제 해결

### 포트가 이미 사용 중일 때

다른 포트 사용:
```bash
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

### 이미지 크기 확인

```bash
docker images | grep ai-image-detector
```

### 불필요한 이미지/컨테이너 정리

```bash
# 중지된 컨테이너 제거
docker container prune

# 사용하지 않는 이미지 제거
docker image prune

# 모든 것 정리 (주의!)
docker system prune -a
```

## 📊 이미지 최적화 팁

1. **Multi-stage 빌드 사용**: 이미 구현되어 있음
2. **.dockerignore 활용**: 불필요한 파일 제외
3. **레이어 캐싱**: requirements.txt를 먼저 복사하여 의존성 변경 시에만 재빌드
4. **알파인 이미지 사용**: 더 작은 이미지 크기 (현재는 slim 사용)

## 🔐 프로덕션 배포

### 보안 고려사항

1. **읽기 전용 볼륨**: `:ro` 플래그 사용
2. **비root 사용자**: Dockerfile에 `USER` 추가
3. **환경 변수**: 민감한 정보는 환경 변수로 관리
4. **네트워크 격리**: Docker 네트워크 사용

### 예시 프로덕션 설정

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  api:
    build:
      context: .
      dockerfile: Dockerfile.api
    ports:
      - "8000:8000"
    environment:
      - PYTHONUNBUFFERED=1
      - LOG_LEVEL=INFO
    restart: always
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
    networks:
      - backend
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

## 📝 체크리스트

- [x] Dockerfile 작성
- [x] .dockerignore 작성
- [x] docker-compose.yml 작성
- [x] Multi-stage 빌드 구현
- [x] 헬스 체크 설정
- [x] 볼륨 마운트 설정
- [x] 네트워크 설정

## 🔗 참고 자료

- [Docker 공식 문서](https://docs.docker.com/)
- [Docker Compose 문서](https://docs.docker.com/compose/)
- [Best Practices for Dockerfile](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/)

