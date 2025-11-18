# 실행 스크립트

이 폴더에는 프로젝트 실행을 위한 스크립트들이 포함되어 있습니다.

## 스크립트 목록

### 학습 스크립트
- `run_cnn_training.py`: CNN 모델 학습 실행
- `run_vit_training.py`: Vision Transformer 모델 학습 실행

### 실행 스크립트
- `run_streamlit.sh`: Streamlit 웹 데모 실행
- `run_api.sh`: FastAPI 백엔드 서버 실행

## 사용 방법

### CNN 모델 학습
```bash
python scripts/run_cnn_training.py
```

### ViT 모델 학습
```bash
python scripts/run_vit_training.py
```

### Streamlit 웹 데모 실행
```bash
bash scripts/run_streamlit.sh
```

### FastAPI 서버 실행
```bash
bash scripts/run_api.sh
```

## 참고

- 학습 스크립트는 `configs/` 폴더의 설정 파일을 사용합니다.
- 실행 전에 필요한 의존성이 설치되어 있는지 확인하세요.
- 자세한 설치 및 실행 방법은 `docs/INSTALL_AND_RUN.md`를 참조하세요.
