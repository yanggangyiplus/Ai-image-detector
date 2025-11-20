# AI Image Detector - 웹 데모 및 API

이 디렉토리에는 AI 이미지 탐지 시스템의 웹 데모와 API 서버가 포함되어 있습니다.

## 📁 파일 구조

```
app/
├── web_demo.py      # Streamlit 웹 데모
├── api.py           # FastAPI 백엔드 API
├── templates/       # HTML 템플릿 (선택사항)
├── static/          # 정적 파일 (CSS, JS 등)
└── README.md        # 이 파일
```

## 🚀 빠른 시작

### 1. Streamlit 웹 데모 실행

```bash
# 프로젝트 루트에서 실행
streamlit run app/web_demo.py

# 또는
cd app
streamlit run web_demo.py
```

웹 브라우저에서 `http://localhost:8501`로 접속하세요.

### 2. FastAPI 서버 실행

```bash
# 프로젝트 루트에서 실행
python app/api.py

# 또는 uvicorn 직접 사용
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
```

API 문서는 `http://localhost:8000/docs`에서 확인할 수 있습니다.

## 📋 기능

### Streamlit 웹 데모 (`web_demo.py`)

- ✅ 이미지 업로드 및 미리보기
- ✅ CNN/ViT 모델 선택
- ✅ 실시간 예측 결과 표시
- ✅ 확률 분포 시각화 (Plotly)
- ✅ 상세 정보 표시
- ✅ 반응형 레이아웃

### FastAPI 백엔드 (`api.py`)

- ✅ RESTful API 엔드포인트
- ✅ 단일 이미지 예측 (`/predict`)
- ✅ 배치 이미지 예측 (`/predict/batch`)
- ✅ 헬스 체크 (`/health`)
- ✅ 모델 정보 조회 (`/models`)
- ✅ 자동 API 문서 (Swagger UI)
- ✅ CORS 지원

## 🔧 API 사용법

### 단일 이미지 예측

```bash
# cURL 사용
curl -X POST "http://localhost:8000/predict?model_type=cnn" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@image.jpg"

# Python requests 사용
import requests

url = "http://localhost:8000/predict"
files = {"file": open("image.jpg", "rb")}
params = {"model_type": "cnn"}
response = requests.post(url, files=files, params=params)
print(response.json())
```

### 배치 이미지 예측

```python
import requests

url = "http://localhost:8000/predict/batch"
files = [
    ("files", open("image1.jpg", "rb")),
    ("files", open("image2.jpg", "rb"))
]
params = {"model_type": "vit"}
response = requests.post(url, files=files, params=params)
print(response.json())
```

### 헬스 체크

```bash
curl http://localhost:8000/health
```

## 📊 API 응답 형식

### 예측 응답 (`/predict`)

```json
{
  "image_path": "/tmp/image.jpg",
  "predicted_class": "Real",
  "predicted_class_idx": 0,
  "confidence": 0.9876,
  "probabilities": {
    "Real": 0.9876,
    "AI": 0.0124
  },
  "is_ai": false,
  "model_type": "CNN"
}
```

### 배치 예측 응답 (`/predict/batch`)

```json
[
  {
    "image_path": "/tmp/image1.jpg",
    "predicted_class": "Real",
    ...
  },
  {
    "image_path": "/tmp/image2.jpg",
    "predicted_class": "AI",
    ...
  }
]
```

## 🛠️ 필요한 패키지

```bash
pip install streamlit fastapi uvicorn python-multipart plotly pillow
```

또는 프로젝트 루트의 `requirements.txt`를 사용하세요:

```bash
pip install -r requirements.txt
```

## 📝 환경 변수

필요시 다음 환경 변수를 설정할 수 있습니다:

- `MODEL_PATH`: 모델 체크포인트 경로 (기본값: `experiments/checkpoints/`)
- `API_HOST`: API 서버 호스트 (기본값: `0.0.0.0`)
- `API_PORT`: API 서버 포트 (기본값: `8000`)

## 🐛 문제 해결

### 모델이 로드되지 않을 때

1. 체크포인트 파일이 존재하는지 확인:
   ```bash
   ls experiments/checkpoints/
   ```

2. 모델 학습이 완료되었는지 확인

### 포트가 이미 사용 중일 때

다른 포트를 사용하세요:

```bash
# Streamlit
streamlit run app/web_demo.py --server.port 8502

# FastAPI
uvicorn app.api:app --port 8001
```

### 메모리 부족 오류

배치 크기를 줄이거나 CPU 모드로 실행하세요.

## 📚 추가 리소스

- [Streamlit 문서](https://docs.streamlit.io/)
- [FastAPI 문서](https://fastapi.tiangolo.com/)
- 프로젝트 메인 README: `../README.md`



