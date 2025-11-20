# AI Image Detector

딥러닝 기반 AI 생성 이미지 탐지 시스템

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 프로젝트 미리보기

![demo](assets/demo_preview.gif)

*Streamlit 웹 데모 화면 - 실시간 AI 생성 이미지 탐지*

## 핵심 성과 요약

| 항목 | 성과 |
|:---:|:---:|
| **성능** | ViT Test Accuracy **97.06%** (CNN 대비 +0.74%p) |
| **Latency** | CNN-FP16 **12ms** (실시간 추론 가능) |
| **데이터셋** | **70,190장** 전처리 + EDA 직접 수행 |
| **서비스 배포** | HuggingFace Spaces + FastAPI + Docker |
| **구현 범위** | 모든 코드/실험/문서 **100% 직접 구현** |

## 문제 정의 & 해결 목적

생성형 AI의 확산으로 가짜 이미지, 딥페이크, AI 생성 콘텐츠가 SNS, 뉴스, 커머스에서 문제로 부상하고 있습니다. 본 프로젝트는 이러한 위협을 사전에 감지하기 위한 컴퓨터 비전 기반 실시간 탐지 시스템입니다.

CNN과 Vision Transformer 두 가지 아키텍처를 비교하여 사용 목적에 맞는 최적 모델을 제시하며, 실제 서비스에 적용 가능한 수준의 성능(97% 이상 정확도)을 달성했습니다.

## 프로젝트 개요

### 목적
AI 생성 이미지와 실제 이미지를 구분하는 이진 분류 태스크에서 CNN(ResNet18)과 Vision Transformer(ViT-Base) 모델의 성능을 비교하고, 각 모델의 장단점을 분석하여 최적의 모델을 선정합니다.

### 주요 특징
- 두 가지 딥러닝 아키텍처 비교: CNN vs Vision Transformer
- 고성능 모델: Test Accuracy 97% 이상 달성
- 실시간 추론: Streamlit 웹 데모 및 FastAPI 백엔드 제공
- 배포 완료: HuggingFace Spaces에 배포되어 즉시 사용 가능
- 체계적인 실험: EDA, 전처리, 학습, 평가 파이프라인 구축

## 시스템 아키텍처

### 전체 시스템 구조

```
┌───────────────────────────────────────────────────────────────┐
│                        사용자 (User)                            │
└────────────────────────────┬──────────────────────────────────┘
                             │
                             ▼
        ┌──────────────────────────────────────┐
        │     HuggingFace Spaces / Streamlit   │
        │         (웹 인터페이스)                  │
        └──────────────────┬───────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────────┐
        │          FastAPI Backend             │
        │      (RESTful API 서버)               │
        └──────────────────┬───────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────────┐
        │           Inference Engine           │
        │          (모델 로더 & 추론 엔진)          │
        └──────────┬───────────────────┬───────┘
                   │                   │
                   ▼                   ▼
        ┌──────────────────┐  ┌──────────────────┐
        │   CNN (ResNet18) │  │  ViT (ViT-Base)  │
        │   11.7M params   │  │  86.7M params    │
        └──────────────────┘  └──────────────────┘
                   │                   │
                   └─────────┬─────────┘
                             ▼
        ┌──────────────────────────────────────┐
        │         Preprocessing Pipeline       │
        │     (Resize, Normalize, Transform)   │
        └──────────────────────────────────────┘
```

### 학습 파이프라인

```
Raw Data (70,190 images)
    │
    ▼
┌──────────────────────────┐
│   Data Preprocessing     │
│  - Resize (224×224)      │
│  - Denoising             │
│  - Histogram Equalization│
└───────────┬──────────────┘
            │
            ▼
┌─────────────────────────┐
│   Stratified Split      │
│  Train: 49,132 (70%)    │
│  Val:   10,528 (15%)    │
│  Test:  10,530 (15%)    │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│   Model Training        │
│  - CNN / ViT            │
│  - Early Stopping       │
│  - Weighted Loss        │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│   Model Evaluation      │
│  - Metrics Calculation  │
│  - Confusion Matrix     │
│  - Error Analysis       │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│   Model Deployment      │
│  - Checkpoint Save      │
│  - API/Web Demo         │
│  - Docker Container     │
└─────────────────────────┘
```

## 모델/기술 스택

### 기술 스택

| 영역 | 기술 | 선택 이유 |
|------|------|----------|
| **Deep Learning** | PyTorch, torchvision, transformers | PyTorch는 연구 및 프로덕션 모두에서 널리 사용되며, 모델 구현과 실험에 유연함 제공 |
| **Web Framework** | Streamlit, FastAPI | Streamlit은 빠른 프로토타이핑에 적합, FastAPI는 고성능 API 서버 구축에 적합 |
| **Data Processing** | PIL, OpenCV, NumPy, Pandas | 이미지 처리와 데이터 조작에 표준 라이브러리 사용 |
| **Visualization** | Matplotlib, Plotly | 실험 결과 시각화 및 대시보드 구축 |
| **Deployment** | Docker, HuggingFace Spaces | Docker는 일관된 배포 환경 제공, HuggingFace Spaces는 모델 공유 및 데모에 적합 |

### 모델 아키텍처

**CNN (ResNet18)**:
- 11.7M 파라미터, ImageNet 사전 학습
- 지역적 특징 학습, 빠른 추론 속도 (15ms)
- ResNet18 선택 이유: ResNet50 대비 2배 빠른 추론 속도로 효율성과 성능의 균형 제공

**ViT (ViT-Base)**:
- 86.7M 파라미터, ImageNet-21k 사전 학습
- 전역적 컨텍스트 이해, 높은 정확도 (97.06%)
- ViT-Base 선택 이유: ViT-Small은 성능 부족, ViT-Large는 비용 과다로 최적 균형점

## 실험 결과

### 데이터셋
- **총 데이터**: 70,190개 이미지 (Train: 49,132 / Val: 10,528 / Test: 10,530)
- **클래스 불균형**: Real:Fake 비율 약 6:1
- **전처리**: Resize (224×224), Denoising, Histogram Equalization

### 모델 비교

| 특징 | CNN (ResNet18) | ViT (ViT-Base) |
|------|----------------|----------------|
| **파라미터 수** | 11.7M | 86.7M |
| **Test Accuracy** | 96.32% | **97.06%** |
| **AI F1 Score** | 87.42% | **89.70%** |
| **Inference Latency (GPU)** | **15ms** | 45ms |
| **Model Size** | **11.7M** | 86.7M |

### 핵심 결론
1. **성능 우위**: ViT가 모든 정확도 지표에서 우수 (약 0.7%p 향상)
2. **속도 우위**: CNN이 추론 속도에서 3배 빠름 (15ms vs 45ms)
3. **권장 사용**: 
   - **정확도 우선**: ViT 모델 사용
   - **실시간 처리**: CNN 모델 사용
   - **균형**: CNN + FP16 양자화 (12ms, 96.28% 정확도)

### 모델 비교 시각화

![Model Comparison](experiments/results/model_comparison_summary.png)

*CNN vs ViT 성능 비교 요약: Accuracy, F1 Score, Confusion Matrix 통합 비교*

**상세 그래프**:
- [CNN 학습 곡선](experiments/results/CNN_resnet18_training_curves.png)
- [ViT 학습 곡선](experiments/results/ViT_vit_base_training_curves.png)
- [CNN Confusion Matrix](experiments/results/CNN_resnet18_test_confusion_matrix.png)
- [ViT Confusion Matrix](experiments/results/ViT_vit_base_test_confusion_matrix.png)

### Ablation Study 결과

| 실험 번호 | 데이터 증강 | Histogram Equalization | Weighted Loss | Test Accuracy | 개선 효과 |
|----------|----------|------------------------|---------------|---------------|---------|
| **Baseline** | - | - | - | 94.2% | - |
| **Exp 2** | - | O | - | 95.8% | +1.6%p |
| **Final** | O | O | O | **96.32%** | **+2.12%p** |

**핵심 발견사항**:
- **Histogram Equalization**: 가장 큰 개선 효과 (+1.6%p) - 다양한 조명 조건 대응
- **데이터 증강**: +0.9%p (RandomHorizontalFlip + RandomRotation)
- **Weighted Loss**: 클래스 불균형 대응 (+0.6%p)
- **시너지 효과**: 개별 요소의 합보다 큰 개선 (+2.12%p)

상세 Ablation Study는 [docs/ABLATION_STUDY.md](docs/ABLATION_STUDY.md)를 참조하세요.

## 핵심 기술 설명

### Design Decision

1. **ResNet18 선택**: 효율성과 성능의 균형 (ResNet50 대비 2배 빠른 추론)
2. **ViT-Base 선택**: 성능과 효율의 최적 균형 (ViT-Small은 성능 부족, ViT-Large는 비용 과다)
3. **Histogram Equalization**: 다양한 조명 조건 대응 (+1.6%p 성능 향상)
4. **Weighted Loss**: 클래스 불균형 해결 (AI F1 +2.1%p 향상)

상세 설명은 [실험 보고서](experiments/reports/experiment_report.md)를 참조하세요.

## 실행 방법

### Quick Start

```bash
# 저장소 클론
git clone https://github.com/yanggangyiplus/Ai-image-detector.git
cd Ai-image-detector

# 의존성 설치
pip install -r requirements.txt

# 웹 데모 실행
streamlit run app/web_demo.py

# 또는 API 서버 실행
uvicorn app.api:app --reload
```

**실행 시간**: 약 10초 내 완료 가능

### 상세 설치 가이드

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

### 배포

#### HuggingFace Spaces 배포

**배포 URL**: [https://huggingface.co/spaces/yanggangyi/Ai-image-detector](https://huggingface.co/spaces/yanggangyi/Ai-image-detector)

**사용 방법**:
1. 위 링크를 클릭하여 Space 페이지 접속
2. 사이드바에서 이미지 업로드
3. "예측하기" 버튼 클릭
4. 실시간으로 AI 생성 이미지 여부 확인

#### 로컬 Docker 배포

```bash
cd deployment/docker
docker-compose up -d
```

자세한 배포 방법은 [deployment/DEPLOYMENT.md](deployment/DEPLOYMENT.md)를 참조하세요.

## 사용 시나리오 (Use Cases)

### 1. SNS 플랫폼 AI 콘텐츠 필터링
**플랫폼**: Instagram, KakaoView, Facebook  
**해결책**: 이미지 업로드 시 실시간 탐지 API 호출, AI 생성 이미지 자동 태깅  
**효과**: 사용자 투명성 향상, 가짜 뉴스/정보 확산 방지

### 2. 쇼핑 플랫폼 제품 이미지 진위 판별
**플랫폼**: 쇼핑몰, 전자상거래 사이트  
**해결책**: 상품 등록 시 이미지 자동 검증, AI 생성 이미지 사용 시 경고 또는 거부  
**효과**: 사기 거래 방지, 소비자 보호 강화

### 3. 뉴스·언론사 AI 생성 이미지 자동 태깅
**플랫폼**: 뉴스 웹사이트, 언론사 CMS  
**해결책**: 기사 작성 시 이미지 자동 검증, AI 생성 이미지 자동 태깅  
**효과**: 가짜 뉴스 방지, 언론 신뢰도 유지

### 4. 중고 거래 플랫폼 사기 방지
**플랫폼**: 번개장터, 당근마켓, 중고나라  
**해결책**: 상품 등록 시 이미지 자동 검증, AI 생성 이미지 탐지 시 경고 표시  
**효과**: 사기 거래 사전 차단, 사용자 피해 방지

## 한계 & 개선 방향

### 현재 한계

- **Diffusion 모델 대응**: GAN 기반 생성 이미지에 비해 diffusion 기반 AI 이미지(Stable Diffusion, DALL-E 등) 탐지가 더 어려움
- **저해상도 이미지**: Resolution이 낮은 이미지(128×128 이하)에서 성능 하락
- **Domain Shift**: SNS 필터, 보정, 압축 등 후처리가 가해진 이미지에서 F1 Score 감소
- **실시간 처리 한계**: ViT 모델의 경우 배치 처리 환경에서만 최적 성능 발휘

### 개선 방향

- **멀티모달 확장**: CLIP backbone 기반 텍스트-이미지 결합 탐지 모델 개발
- **전처리 강화**: SAM(Segment Anything Model) 기반 segmentation pre-processing으로 ROI 추출
- **데이터셋 확장**: Diffusion-based Fake Image 생성 도구 추가 및 학습 데이터 보강
- **모델 보정**: Post-Training Calibration (Temperature Scaling) 적용으로 신뢰도 보정
- **앙상블 모델**: CNN과 ViT의 앙상블을 통한 성능 향상 연구
- **경량화**: MobileNet 기반 경량 모델 개발로 모바일 환경 대응

## 개인 기여도

이 프로젝트는 **개인 프로젝트**로, 모든 작업을 직접 수행했습니다.

### 데이터 수집 및 전처리
- 데이터 수집: 3개 데이터셋 통합 (총 70,190개 이미지)
- 데이터 전처리: Resize, Denoising, Histogram Equalization 구현
- 데이터 분할: Stratified Split으로 클래스 비율 유지하며 Train/Val/Test 분할
- EDA: 탐색적 데이터 분석 및 시각화

### 모델 설계 및 실험
- 모델 선택: CNN (ResNet18) 및 ViT (ViT-Base) 아키텍처 선택 및 구현
- 실험 설계: 하이퍼파라미터 튜닝 전략 수립
- Ablation Study: 데이터 증강, 전처리, 손실 함수 효과 분석
- 하이퍼파라미터 튜닝: Batch Size, Learning Rate, Optimizer 최적화

### 학습 파이프라인 구축
- 학습 코드: PyTorch 기반 학습 파이프라인 구현
- 평가 시스템: Accuracy, Precision, Recall, F1, Confusion Matrix 구현
- Early Stopping: 과적합 방지 메커니즘 구현
- 체크포인트 관리: Best/Latest 모델 자동 저장

### 웹 애플리케이션 개발
- Streamlit UI: 웹 데모 인터페이스 설계 및 구현
- FastAPI 백엔드: RESTful API 서버 구축
- 프론트엔드-백엔드 연결: API 통신 및 에러 처리 구현
- 사용자 경험: 이미지 업로드, 결과 시각화, 확률 분포 표시

### 배포 및 인프라
- Docker: Dockerfile 작성 및 Docker Compose 구성
- HuggingFace Spaces: 배포 환경 구성 및 배포 수행
- 배포 문제 해결: 403 에러, 파일 시스템 권한 등 이슈 해결
- CI/CD: GitHub Actions 워크플로우 설계 (향후 구현 예정)

### 문서화
- README: 문서 직접 작성 후 LLM 정리 도움 받음 (초안 작성 및 구조 설계는 본인 수행)
- 실험 보고서: 상세한 실험 분석 및 결과 해석
- 코드 주석: 주요 함수 및 클래스에 한국어 주석 추가
- 배포 가이드: Docker, HuggingFace Spaces 배포 가이드 작성

## 프로젝트 구조

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

## 라이선스 & 작성자

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

**작성자**: yanggangyi

- GitHub: [@yanggangyiplus](https://github.com/yanggangyiplus)
- HuggingFace: [@yanggangyi](https://huggingface.co/yanggangyi)

## 상세 문서

### 핵심 문서
- [실험 보고서](experiments/reports/experiment_report.md) - 상세한 실험 분석 및 결과 해석
- [배포 가이드](deployment/DEPLOYMENT.md) - Docker, HuggingFace Spaces 배포 방법

### 추가 문서
- [Ablation Study 상세](docs/ABLATION_STUDY.md) - 전체 Ablation Study 결과
- [모델 Explainability](docs/EXPLAINABILITY.md) - Grad-CAM, Attention Map 상세 설명
- [프로덕션 배포](docs/PRODUCTION.md) - Kubernetes, MLflow, CI/CD, 모니터링
- [보안 및 윤리](docs/ETHICS.md) - 모델 오탐 최소화, 편향 대응, 민감 이미지 처리
- [설치 및 실행 가이드](docs/INSTALL_AND_RUN.md) - 상세한 설치 및 실행 방법
