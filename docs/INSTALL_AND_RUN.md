# 설치 및 실행 가이드

## 1. 필요한 패키지 설치

```bash
pip install streamlit fastapi uvicorn plotly python-multipart
```

또는

```bash
pip install -r requirements.txt
```

## 2. Streamlit 웹 데모 실행

```bash
# 방법 1: 스크립트 사용
./run_streamlit.sh

# 방법 2: 직접 실행
streamlit run app/web_demo.py

# 방법 3: 포트 지정
streamlit run app/web_demo.py --server.port 8501
```

브라우저에서 `http://localhost:8501` 접속

## 3. FastAPI 서버 실행

```bash
# 방법 1: 스크립트 사용
./run_api.sh

# 방법 2: 직접 실행
python app/api.py

# 방법 3: uvicorn 직접 사용
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
```

- API 문서: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 4. 테스트

### Streamlit 데모 테스트
1. 브라우저에서 http://localhost:8501 접속
2. 사이드바에서 모델 선택 (CNN 또는 ViT)
3. 이미지 업로드
4. "예측하기" 버튼 클릭

### FastAPI 테스트
```bash
# 헬스 체크
curl http://localhost:8000/health

# 이미지 예측 (예시)
curl -X POST "http://localhost:8000/predict?model_type=cnn" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@data/test/real/dataset_1_processed_1604_3.jpg"
```
