# 🖼️ AI Image Detector

딥러닝 기반 AI 생성 이미지 탐지 시스템

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 목차

- [프로젝트 개요](#-프로젝트-개요)
- [데이터셋 설명](#-데이터셋-설명)
- [모델 설명](#-모델-설명)
- [실험 결과](#-실험-결과)
- [데모 및 배포](#-데모-및-배포)
- [프로젝트 회고](#-프로젝트-회고)
- [설치 및 실행](#-설치-및-실행)
- [프로젝트 구조](#-프로젝트-구조)

---

## 🎯 프로젝트 개요

### 목적
AI 생성 이미지와 실제 이미지를 구분하는 이진 분류 태스크에서 **CNN(ResNet18)**과 **Vision Transformer(ViT-Base)** 모델의 성능을 비교하고, 각 모델의 장단점을 분석하여 최적의 모델을 선정합니다.

### 주요 특징
- ✅ **두 가지 딥러닝 아키텍처 비교**: CNN vs Vision Transformer
- ✅ **고성능 모델**: Test Accuracy 97% 이상 달성
- ✅ **실시간 추론**: Streamlit 웹 데모 및 FastAPI 백엔드 제공
- ✅ **배포 완료**: HuggingFace Spaces에 배포되어 즉시 사용 가능
- ✅ **체계적인 실험**: EDA, 전처리, 학습, 평가 파이프라인 구축

### 기술 스택
- **Deep Learning**: PyTorch, torchvision, transformers
- **Web Framework**: Streamlit, FastAPI
- **Data Processing**: PIL, OpenCV, NumPy, Pandas
- **Visualization**: Matplotlib, Plotly
- **Deployment**: Docker, HuggingFace Spaces

---

## 📊 데이터셋 설명

### 데이터셋 구성
- **총 데이터**: 70,190개 이미지
  - **Train**: 49,132개 (Real: 42,099개, Fake: 7,033개)
  - **Validation**: 10,528개 (Real: 9,021개, Fake: 1,507개)
  - **Test**: 10,530개 (Real: 9,022개, Fake: 1,508개)

### 데이터셋 특성
- **클래스 불균형**: Real:Fake 비율 약 **6:1**
- **이미지 크기**: 224×224로 전처리
- **데이터 소스**: 
  - Dataset 1: CIFAKE 데이터셋
  - Dataset 2: AI 생성 이미지 및 실제 이미지
  - Dataset 3: 다양한 해상도의 이미지

### 데이터 전처리
1. **이미지 리사이징**: 224×224로 통일
2. **노이즈 제거**: Non-local Means Denoising 적용
3. **색상 정규화**: Histogram Equalization
4. **데이터 증강**: 학습 시 Random Crop, Horizontal Flip, Color Jitter 적용

### 데이터 불균형 해결
- **Stratified Split**: 클래스 비율 유지하며 데이터 분할
- **Weighted Loss**: 클래스 불균형을 고려한 손실 함수 사용
- **데이터 통합**: 3개 데이터셋을 통합하여 대형 데이터셋 구축

---

## 🤖 모델 설명

### 1. CNN (ResNet18)

#### 아키텍처
- **모델**: ResNet18 (11.7M 파라미터)
- **백본**: ImageNet 사전 학습된 ResNet18
- **분류기**: 512차원 FC 레이어 → 2차원 출력

#### 특징
- ✅ **경량 모델**: 빠른 추론 속도
- ✅ **지역적 특징 학습**: 컨볼루션 필터를 통한 지역 패턴 인식
- ✅ **안정적인 성능**: 검증된 아키텍처

#### 학습 설정
- **배치 크기**: 32
- **학습률**: 1e-4
- **Optimizer**: AdamW
- **Scheduler**: Cosine Annealing
- **Best Epoch**: 6 (Early Stopping)

### 2. Vision Transformer (ViT-Base)

#### 아키텍처
- **모델**: ViT-Base (86.7M 파라미터)
- **백본**: HuggingFace transformers의 `google/vit-base-patch16-224`
- **패치 크기**: 16×16
- **어텐션 헤드**: 12개

#### 특징
- ✅ **글로벌 컨텍스트 이해**: Self-Attention 메커니즘
- ✅ **전역적 패턴 학습**: 이미지 전체의 관계를 동시에 학습
- ✅ **고성능**: 더 높은 정확도 달성

#### 학습 설정
- **배치 크기**: 16
- **학습률**: 1e-5 (Fine-tuning)
- **Optimizer**: AdamW
- **Scheduler**: ReduceLROnPlateau
- **Best Epoch**: 5 (Early Stopping)

### 모델 구조 비교

| 특징 | CNN (ResNet18) | ViT (ViT-Base) |
|------|----------------|----------------|
| **파라미터 수** | 11.7M | 86.7M |
| **학습 방식** | 지역적 특징 학습 | 전역적 컨텍스트 학습 |
| **어텐션** | ❌ | ✅ Self-Attention |
| **추론 속도** | 빠름 | 상대적으로 느림 |
| **메모리 사용량** | 적음 | 많음 |

---

## 📈 실험 결과

### 성능 지표 비교

| 지표 | CNN (ResNet18) | ViT (ViT-Base) | 차이 |
|------|----------------|----------------|------|
| **Test Accuracy** | 96.32% | **97.06%** | +0.74%p |
| **Test Precision** | 96.40% | **97.05%** | +0.65%p |
| **Test Recall** | 96.32% | **97.06%** | +0.74%p |
| **Test F1 Score** | 96.35% | **97.05%** | +0.70%p |
| **Best Val Accuracy** | 96.42% | **97.28%** | +0.86%p |
| **Best Val Loss** | 0.0919 | **0.0736** | -0.0183 |

### 클래스별 성능 분석

#### CNN (ResNet18)
- **Real 클래스**: Precision 98.18%, Recall 97.52%, F1 97.85%
- **AI 클래스**: Precision 85.72%, Recall 89.19%, F1 87.42%

#### ViT (ViT-Base)
- **Real 클래스**: Precision 98.25%, Recall 98.32%, F1 98.28%
- **AI 클래스**: Precision 89.88%, Recall 89.52%, F1 89.70%

### 학습 곡선

#### CNN 학습 곡선
- **Train Loss**: 0.184 → 0.025 (16 epochs)
- **Train Accuracy**: 92.6% → 99.1%
- **Val Loss**: 0.231 → 0.092 (Best: 0.092 at epoch 6)
- **Val Accuracy**: 90.4% → 96.4% (Best: 96.4% at epoch 6)

#### ViT 학습 곡선
- **Train Loss**: 0.161 → 0.003 (15 epochs)
- **Train Accuracy**: 93.9% → 99.9%
- **Val Loss**: 0.144 → 0.074 (Best: 0.074 at epoch 5)
- **Val Accuracy**: 94.6% → 97.3% (Best: 97.3% at epoch 5)

### Confusion Matrix

#### CNN (ResNet18)
```
                Predicted
              Real    AI
Actual Real   8,798   224
       AI      163   1,345
```

#### ViT (ViT-Base)
```
                Predicted
              Real    AI
Actual Real   8,870   152
       AI      158   1,350
```

### 주요 발견사항

1. **ViT가 모든 지표에서 우수한 성능**을 보임 (약 0.7%p 향상)
2. **두 모델 모두 Real 클래스에서 높은 성능** (약 98%)
3. **AI 클래스 탐지에서 ViT가 더 우수** (F1: 87.42% → 89.70%)
4. **False Positive 감소**: Real을 AI로 오분류한 경우 ViT가 224개 → 152개로 감소 (32% 개선)

### 성능 그래프

실험 결과 그래프는 `experiments/results/` 폴더에서 확인할 수 있습니다:
- `CNN_resnet18_test_confusion_matrix.png`
- `ViT_vit_base_test_confusion_matrix.png`

---

## 🚀 데모 및 배포

### 🌐 HuggingFace Spaces 배포

**배포 URL**: [https://huggingface.co/spaces/yanggangyi/Ai-image-detector](https://huggingface.co/spaces/yanggangyi/Ai-image-detector)

#### 사용 방법
1. 위 링크를 클릭하여 Space 페이지 접속
2. 사이드바에서 이미지 업로드
3. "예측하기" 버튼 클릭
4. 실시간으로 AI 생성 이미지 여부 확인

#### 배포된 기능
- ✅ CNN 모델 실시간 추론
- ✅ 확률 분포 시각화
- ✅ 상세 예측 정보 제공

### 💻 로컬 실행

#### Streamlit 웹 데모
```bash
bash scripts/run_streamlit.sh
```
또는
```bash
streamlit run app/web_demo.py --server.port 8501
```

#### FastAPI 백엔드
```bash
bash scripts/run_api.sh
```
또는
```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
```

### 🐳 Docker 배포

```bash
cd deployment/docker
docker-compose up -d
```

자세한 배포 방법은 [deployment/DEPLOYMENT.md](deployment/DEPLOYMENT.md)를 참조하세요.

---

## 💭 프로젝트 회고

### 성공한 점

1. **체계적인 실험 설계**
   - EDA를 통한 데이터 특성 파악
   - 두 가지 아키텍처의 체계적 비교
   - 재현 가능한 실험 환경 구축

2. **높은 성능 달성**
   - Test Accuracy 97% 이상 달성
   - 클래스 불균형 문제 해결
   - 안정적인 모델 성능

3. **완전한 배포 파이프라인**
   - Streamlit 웹 데모
   - FastAPI 백엔드 API
   - HuggingFace Spaces 배포
   - Docker 컨테이너화

4. **코드 품질**
   - 모듈화된 코드 구조
   - 재사용 가능한 컴포넌트
   - 상세한 문서화

### 어려웠던 점 및 해결 과정

1. **데이터 불균형 문제**
   - **문제**: Real:Fake 비율이 6:1로 심각한 불균형
   - **해결**: 
     - Stratified Split으로 클래스 비율 유지
     - Weighted Loss 함수 사용
     - 데이터셋 통합으로 샘플 수 증가

2. **HuggingFace Spaces 배포 시 403 에러**
   - **문제**: 이미지 업로드 시 AxiosError 403 발생
   - **해결**:
     - Streamlit 설정 파일 추가 (`.streamlit/config.toml`)
     - CORS 및 XSRF 보호 비활성화
     - PIL Image 객체를 직접 사용하여 임시 파일 저장 제거

3. **모델 용량 제한**
   - **문제**: HuggingFace Spaces 1GB 제한으로 ViT 모델 업로드 불가
   - **해결**: CNN 모델만 배포 (ViT는 로컬에서 사용 가능)

4. **학습 시간 및 리소스**
   - **문제**: ViT 모델 학습에 많은 시간과 메모리 필요
   - **해결**: 
     - 배치 크기 조정 (32 → 16)
     - Early Stopping 적용
     - 학습률 조정 (1e-5)

### 배운 점

1. **모델 선택의 중요성**
   - ViT가 CNN보다 약 0.7%p 높은 성능을 보였지만, 파라미터 수는 7배 이상
   - 실제 서비스에서는 추론 속도와 정확도의 트레이드오프 고려 필요

2. **데이터 전처리의 중요성**
   - 체계적인 전처리 파이프라인 구축이 모델 성능에 큰 영향
   - 데이터 불균형 해결이 핵심

3. **배포 환경의 차이**
   - 로컬 환경과 클라우드 환경의 차이 이해
   - 파일 시스템 권한, 용량 제한 등 고려 필요

### 향후 개선 방향

1. **모델 성능 향상**
   - 더 큰 모델 (ViT-Large) 실험
   - 앙상블 모델 구축
   - Knowledge Distillation 적용

2. **데이터셋 확장**
   - 더 다양한 AI 생성 이미지 추가
   - 데이터 불균형 완전 해결
   - 데이터 증강 기법 개선

3. **실시간 성능 최적화**
   - 모델 양자화 (Quantization)
   - ONNX 변환으로 추론 속도 향상
   - 배치 추론 최적화

4. **사용자 경험 개선**
   - 더 직관적인 UI/UX
   - 배치 처리 기능 추가
   - 예측 결과 시각화 개선

---

## 🛠️ 설치 및 실행

### 요구사항
- Python 3.11+
- CUDA 지원 GPU (선택사항, CPU도 가능)

### 설치

1. **저장소 클론**
```bash
git clone https://github.com/yanggangyiplus/Ai-image-detector.git
cd Ai-image-detector
```

2. **의존성 설치**
```bash
pip install -r requirements.txt
```

3. **데이터 준비**
   - 데이터셋을 `data/raw/` 폴더에 배치
   - 전처리 스크립트 실행: `notebooks/preprocessing.ipynb`

### 모델 학습

#### CNN 모델 학습
```bash
python scripts/run_cnn_training.py
```

#### ViT 모델 학습
```bash
python scripts/run_vit_training.py
```

### 추론

#### 단일 이미지 추론
```bash
python examples/single_image_inference.py --image_path path/to/image.jpg
```

#### 배치 추론
```bash
python examples/batch_inference.py --input_dir path/to/images/
```

자세한 사용 방법은 [docs/INSTALL_AND_RUN.md](docs/INSTALL_AND_RUN.md)를 참조하세요.

---

## 📁 프로젝트 구조

```
Ai-image-detector/
├── app/                    # 웹 애플리케이션
│   ├── web_demo.py        # Streamlit 웹 데모
│   ├── api.py             # FastAPI 백엔드
│   └── templates/         # HTML 템플릿
│
├── configs/               # 설정 파일
│   ├── config_cnn.yaml
│   ├── config_vit.yaml
│   └── config_eval.yaml
│
├── data/                  # 데이터
│   ├── raw/              # 원본 데이터
│   ├── processed/        # 전처리된 데이터
│   ├── train/            # 학습 데이터
│   ├── val/              # 검증 데이터
│   └── test/             # 테스트 데이터
│
├── deployment/            # 배포 관련
│   ├── docker/           # Docker 파일
│   └── huggingface/      # HuggingFace Spaces 배포
│
├── experiments/           # 실험 결과
│   ├── checkpoints/      # 모델 체크포인트
│   ├── logs/            # 학습 로그
│   ├── results/         # 결과 그래프
│   └── reports/         # 실험 보고서
│
├── notebooks/            # Jupyter 노트북
│   ├── EDA.ipynb        # 탐색적 데이터 분석
│   ├── preprocessing.ipynb
│   ├── model_CNN.ipynb
│   ├── model_ViT.ipynb
│   └── model_compare.ipynb
│
├── scripts/              # 실행 스크립트
│   ├── run_cnn_training.py
│   ├── run_vit_training.py
│   ├── run_streamlit.sh
│   └── run_api.sh
│
├── src/                  # 소스 코드
│   ├── data/            # 데이터 처리
│   ├── models/          # 모델 정의
│   ├── training/        # 학습 관련
│   ├── inference/       # 추론 관련
│   └── utils/           # 유틸리티
│
├── examples/            # 예제 코드
├── docs/               # 문서
├── README.md           # 프로젝트 설명
└── requirements.txt    # 의존성 목록
```

---

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

---

## 👤 작성자

**yanggangyi**

- GitHub: [@yanggangyiplus](https://github.com/yanggangyiplus)
- HuggingFace: [@yanggangyi](https://huggingface.co/yanggangyi)

---

## 🙏 감사의 말

- PyTorch 팀
- HuggingFace 팀
- Streamlit 팀
- 오픈소스 커뮤니티

---

## 📚 참고 자료

- [실험 보고서](experiments/reports/experiment_report.md)
- [배포 가이드](deployment/DEPLOYMENT.md)
- [설치 및 실행 가이드](docs/INSTALL_AND_RUN.md)

