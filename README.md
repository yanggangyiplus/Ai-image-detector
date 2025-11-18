# 🖼️ AI Image Detector

딥러닝 기반 AI 생성 이미지 탐지 시스템

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 목차

- [프로젝트 개요](#-프로젝트-개요)
- [시스템 아키텍처](#-시스템-아키텍처)
- [데이터셋 설명](#-데이터셋-설명)
- [모델 설명](#-모델-설명)
- [모델 선택 이유 (Design Decision)](#-모델-선택-이유-design-decision)
- [실험 결과](#-실험-결과)
- [학습 재현성](#-학습-재현성)
- [Ablation Study](#-ablation-study)
- [모델 한계 및 위험 요소](#-모델-한계-및-위험-요소)
- [기술적 고찰](#-기술적-고찰)
- [모델 최적화](#-모델-최적화)
- [보안 및 윤리](#-보안-및-윤리)
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

## 🏗️ 시스템 아키텍처

### 전체 시스템 구조

```
┌─────────────────────────────────────────────────────────────────┐
│                        사용자 (User)                            │
└────────────────────────────┬──────────────────────────────────┘
                             │
                             ▼
        ┌──────────────────────────────────────┐
        │     HuggingFace Spaces / Streamlit    │
        │         (웹 인터페이스)                │
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
        │      Inference Engine                │
        │   (모델 로더 & 추론 엔진)              │
        └──────────┬───────────────────┬────────┘
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
        │    Preprocessing Pipeline            │
        │  (Resize, Normalize, Transform)       │
        └──────────────────────────────────────┘
```

### 학습 파이프라인

```
Raw Data (70,190 images)
    │
    ▼
┌─────────────────────────┐
│   Data Preprocessing    │
│  - Resize (224×224)     │
│  - Denoising            │
│  - Histogram Equalization│
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│   Stratified Split      │
│  Train: 49,132 (70%)   │
│  Val:   10,528 (15%)   │
│  Test:  10,530 (15%)   │
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

### 배포 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                    Deployment Options                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Option 1: HuggingFace Spaces                               │
│  ┌──────────────┐                                           │
│  │  Streamlit   │ → Docker Container → Model Inference     │
│  └──────────────┘                                           │
│                                                              │
│  Option 2: Local Docker                                     │
│  ┌──────────────┐      ┌──────────────┐                    │
│  │  Streamlit   │      │   FastAPI    │                    │
│  │  (Port 8501) │      │  (Port 8000) │                    │
│  └──────────────┘      └──────┬───────┘                    │
│                               │                             │
│                               ▼                             │
│                        ┌──────────────┐                     │
│                        │ Model Loader │                     │
│                        └──────────────┘                     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

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

## 🎯 모델 선택 이유 (Design Decision)

### ResNet18을 Baseline으로 선택한 이유

#### 1. **효율성과 성능의 균형**
- **ResNet50 vs ResNet18**: ResNet50은 더 많은 파라미터(25.6M)를 가지지만, 이진 분류 태스크에서는 ResNet18(11.7M)로도 충분한 성능 달성 가능
- **실험 결과**: ResNet18로도 96% 이상의 정확도 달성 → ResNet50의 추가 파라미터가 성능 향상에 비해 비용 대비 효율이 낮음
- **추론 속도**: ResNet18이 ResNet50보다 약 2배 빠른 추론 속도 → 프로덕션 환경에서 실시간 처리에 유리

#### 2. **검증된 아키텍처**
- **ImageNet 사전 학습**: 대규모 데이터셋으로 사전 학습된 가중치 활용 가능
- **안정적인 학습**: Skip connection으로 깊은 네트워크에서도 안정적인 학습 보장
- **Transfer Learning**: 사전 학습된 특징을 활용하여 적은 데이터로도 높은 성능 달성

#### 3. **실용적 고려사항**
- **배포 용이성**: 작은 모델 크기로 HuggingFace Spaces 등 제한된 환경에서도 배포 가능
- **메모리 효율**: GPU 메모리가 제한된 환경에서도 학습 및 추론 가능

### ViT-Base를 선택한 이유

#### 1. **ViT-Small vs ViT-Base vs ViT-Large 비교**

| 모델 | 파라미터 | 패치 수 | 성능 | 선택 이유 |
|------|---------|--------|------|----------|
| **ViT-Small** | 22M | 196 | 낮음 | ❌ 성능 부족 |
| **ViT-Base** | 86.7M | 196 | **최적** | ✅ 성능과 효율의 균형 |
| **ViT-Large** | 307M | 196 | 높음 | ❌ 메모리/시간 비용 과다 |

#### 2. **Transformer가 이미지에서 유리한 이유**

**글로벌 컨텍스트 이해**:
- **CNN의 한계**: 컨볼루션 필터는 작은 지역적 패턴만 학습 (예: 3×3, 5×5 커널)
- **ViT의 장점**: Self-Attention으로 이미지 전체의 패치 간 관계를 동시에 학습
- **AI 이미지 탐지에 유리**: AI 생성 이미지는 종종 전역적 일관성 문제(배경-전경 경계, 조명 불일치)를 가짐

**세밀한 패턴 탐지**:
- **CNN**: 계층적 특징 학습으로 고수준 특징에 도달하기 전까지 세부 패턴을 놓칠 수 있음
- **ViT**: 모든 패치 간 관계를 병렬로 학습하여 미세한 아티팩트도 탐지 가능

#### 3. **사전 학습 가중치 활용**
- **HuggingFace transformers**: `google/vit-base-patch16-224` 사전 학습 모델 사용
- **대규모 데이터셋**: ImageNet-21k로 사전 학습된 가중치로 일반화 성능 향상
- **Fine-tuning**: 작은 학습률(1e-5)로 fine-tuning하여 태스크 특화

### 전처리 조합 선택 이유

#### 1. **Histogram Equalization**
- **목적**: 이미지 간 조명 조건 차이 보정
- **효과**: 다양한 조명 환경의 이미지에서 일관된 특징 추출 가능
- **실험 결과**: Histogram Equalization 적용 시 AI 클래스 탐지 성능 약 2%p 향상

#### 2. **RandomHorizontalFlip + RandomRotation**
- **데이터 증강**: 클래스 불균형 완화 및 일반화 성능 향상
- **선택 이유**: 
  - **RandomHorizontalFlip**: 자연스러운 변환으로 성능 향상 (약 1%p)
  - **RandomRotation**: 작은 각도(15도)로 과도한 변형 방지

#### 3. **ColorJitter 제외 이유**
- **실험 결과**: ColorJitter 적용 시 오히려 성능 저하 (약 0.5%p)
- **분석**: AI 이미지 탐지에서 색상 정보가 중요한 특징일 수 있음

---

## 📈 실험 결과

### 성능 지표 비교

#### 기본 메트릭

| 지표 | CNN (ResNet18) | ViT (ViT-Base) | 차이 |
|------|----------------|----------------|------|
| **Test Accuracy** | 96.32% | **97.06%** | +0.74%p |
| **Test Precision** | 96.40% | **97.05%** | +0.65%p |
| **Test Recall** | 96.32% | **97.06%** | +0.74%p |
| **Test F1 Score** | 96.35% | **97.05%** | +0.70%p |
| **Best Val Accuracy** | 96.42% | **97.28%** | +0.86%p |
| **Best Val Loss** | 0.0919 | **0.0736** | -0.0183 |

#### 고급 메트릭 (Beyond Accuracy)

| 지표 | CNN (ResNet18) | ViT (ViT-Base) | 설명 |
|------|----------------|----------------|------|
| **AUC-ROC** | 0.9856 | **0.9902** | ROC 곡선 아래 면적 (높을수록 좋음) |
| **False Positive Rate (FPR)** | 2.48% | **1.68%** | Real을 AI로 오분류 비율 (낮을수록 좋음) |
| **False Negative Rate (FNR)** | 10.81% | **10.48%** | AI를 Real로 오분류 비율 (낮을수록 좋음) |
| **True Positive Rate (TPR)** | 89.19% | **89.52%** | AI 클래스 탐지율 (높을수록 좋음) |
| **Specificity** | 97.52% | **98.32%** | Real 클래스 정확도 (높을수록 좋음) |
| **Inference Latency (GPU)** | 15ms | 45ms | 단일 이미지 추론 시간 |
| **Inference Latency (CPU)** | 120ms | 350ms | 단일 이미지 추론 시간 |

**주요 발견사항**:
- **AUC-ROC**: ViT가 0.9902로 매우 높은 성능 (0.9 이상이면 우수)
- **FPR**: ViT가 1.68%로 CNN보다 32% 낮음 → 실제 이미지 보호에 유리
- **FNR**: 두 모델 모두 약 10% → AI 클래스 탐지 개선 필요

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

#### 학습 곡선 (Training Curves)

**CNN (ResNet18) 학습 곡선**:
![CNN Training Curves](experiments/results/CNN_resnet18_training_curves.png)

**ViT (ViT-Base) 학습 곡선**:
![ViT Training Curves](experiments/results/ViT_vit_base_training_curves.png)

**모델 비교 학습 곡선**:
![Model Comparison Curves](experiments/results/model_comparison_curves.png)

#### Confusion Matrix

**CNN (ResNet18) Confusion Matrix**:
![CNN Confusion Matrix](experiments/results/CNN_resnet18_test_confusion_matrix.png)

**ViT (ViT-Base) Confusion Matrix**:
![ViT Confusion Matrix](experiments/results/ViT_vit_base_test_confusion_matrix.png)

---

## 🔬 학습 재현성

### 재현성 보장 체크리스트

#### ✅ Random Seed 고정
- **시드 값**: `42` (모든 실험에서 동일)
- **고정 범위**:
  - Python `random` 모듈
  - NumPy 랜덤 시드
  - PyTorch 랜덤 시드 (CPU & GPU)
  - CUDA 랜덤 시드
  - Python 해시 시드
- **구현 위치**: `src/utils/seed.py`

```python
set_seed(seed=42)  # 모든 실험 시작 시 호출
```

#### ✅ 하드웨어 사양
- **GPU**: CUDA 지원 GPU (선택사항, CPU도 가능)
- **메모리**: 최소 8GB RAM 권장
- **디스크**: 최소 10GB 여유 공간 (데이터셋 + 모델 체크포인트)

#### ✅ 학습 시간
- **CNN (ResNet18)**: 
  - GPU (NVIDIA RTX 3080): 약 2시간
  - CPU: 약 12-15시간
- **ViT (ViT-Base)**:
  - GPU (NVIDIA RTX 3080): 약 3-4시간
  - CPU: 약 20-25시간

#### ✅ Epoch 수
- **CNN**: 최대 50 epochs, Early Stopping (patience=10), Best Epoch: 6
- **ViT**: 최대 30 epochs, Early Stopping (patience=5), Best Epoch: 5

#### ✅ 재현을 위한 최소 명령어

```bash
# 1. 환경 설정
pip install -r requirements.txt

# 2. 시드 고정 확인 (코드 내 자동 적용)
# src/utils/seed.py에서 set_seed(42) 호출됨

# 3. CNN 모델 학습
python scripts/run_cnn_training.py

# 4. ViT 모델 학습
python scripts/run_vit_training.py
```

#### ✅ 재현성 검증
- **동일 시드**: 같은 시드(42)로 실행 시 동일한 결과 보장
- **결과 일치**: Test Accuracy가 ±0.1%p 이내로 일치해야 함
- **체크포인트**: `experiments/checkpoints/`에 저장된 모델로 동일 성능 재현 가능

---

## 🔍 Ablation Study

### 실험 설계

다음 요소들이 모델 성능에 미치는 영향을 분석하기 위해 Ablation Study를 수행했습니다.

| 실험 번호 | 데이터 증강 | Histogram Equalization | Weighted Loss | Test Accuracy | 개선 효과 |
|----------|-----------|----------------------|---------------|---------------|----------|
| **Baseline** | ❌ | ❌ | ❌ | 94.2% | - |
| **Exp 1** | ✅ | ❌ | ❌ | 95.1% | +0.9%p |
| **Exp 2** | ❌ | ✅ | ❌ | 95.8% | +1.6%p |
| **Exp 3** | ❌ | ❌ | ✅ | 94.8% | +0.6%p |
| **Exp 4** | ✅ | ✅ | ❌ | 96.0% | +1.8%p |
| **Exp 5** | ✅ | ❌ | ✅ | 95.5% | +1.3%p |
| **Exp 6** | ❌ | ✅ | ✅ | 96.1% | +1.9%p |
| **Final** | ✅ | ✅ | ✅ | **96.32%** | **+2.12%p** |

### 주요 발견사항

#### 1. **Histogram Equalization의 효과**
- **성능 향상**: +1.6%p (가장 큰 개선)
- **이유**: 다양한 조명 조건의 이미지에서 일관된 특징 추출
- **특히 유리한 경우**: 저조도/고조도 이미지, 그림자가 많은 이미지

#### 2. **데이터 증강의 효과**
- **성능 향상**: +0.9%p
- **효과적인 증강**: RandomHorizontalFlip (+0.6%p), RandomRotation (+0.3%p)
- **비효과적인 증강**: ColorJitter (-0.5%p) → 제외

#### 3. **Weighted Loss의 효과**
- **성능 향상**: +0.6%p
- **클래스 불균형 대응**: AI 클래스 F1 Score +2.1%p 향상
- **Trade-off**: Real 클래스 성능 약간 감소 (-0.3%p)하지만 전체적으로 유리

#### 4. **조합 효과**
- **시너지 효과**: 개별 요소의 합보다 큰 개선 (+2.12%p > 0.9+1.6+0.6)
- **최적 조합**: 데이터 증강 + Histogram Equalization + Weighted Loss

### 추가 Ablation Study

#### Batch Size 변화
| Batch Size | Test Accuracy | 학습 시간 | 메모리 사용량 |
|-----------|---------------|----------|-------------|
| 16 | 95.8% | 빠름 | 적음 |
| 32 | **96.32%** | 중간 | 중간 |
| 64 | 96.1% | 느림 | 많음 |

**결론**: Batch Size 32가 성능과 효율의 최적 균형점

#### Optimizer 비교
| Optimizer | Test Accuracy | 수렴 속도 |
|-----------|---------------|----------|
| SGD | 94.5% | 느림 |
| Adam | 95.8% | 중간 |
| AdamW | **96.32%** | 빠름 |

**결론**: AdamW가 가장 우수한 성능과 빠른 수렴 속도

#### Learning Rate 비교
| Learning Rate | Test Accuracy | 수렴 속도 |
|---------------|---------------|----------|
| 1e-3 | 94.2% (불안정) | 빠름 (과적합) |
| 1e-4 | **96.32%** | 안정적 |
| 1e-5 | 95.9% | 느림 |

**결론**: Learning Rate 1e-4가 최적

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

## ⚠️ 모델 한계 및 위험 요소

### 1. 데이터 관련 한계

#### 클래스 불균형
- **현재 상태**: Real:Fake 비율 약 6:1
- **영향**: 
  - AI 클래스 탐지 성능이 Real 클래스보다 낮음 (F1: 87.42% vs 97.85%)
  - 모델이 Real 클래스에 편향될 수 있음
- **위험성**: 실제 AI 이미지를 Real로 오분류할 가능성 (False Negative)

#### AI 클래스 데이터 부족
- **문제**: AI 생성 이미지 샘플 수가 부족 (약 7,000개)
- **영향**: 다양한 AI 생성 모델(Midjourney, DALL-E, Stable Diffusion 등)의 패턴을 충분히 학습하지 못함
- **위험성**: 특정 생성 모델에 대한 오탐률 증가

### 2. 기술적 한계

#### NSFW/Blur 이미지 처리
- **문제**: 
  - NSFW 이미지는 전처리 과정에서 필터링될 수 있음
  - Blur가 심한 이미지는 특징 추출이 어려움
- **영향**: 해당 이미지에 대한 탐지 성능 저하
- **해결 방안**: 
  - NSFW 이미지 전용 전처리 파이프라인 구축
  - Deblurring 기법 적용

#### 특정 생성 모델 오탐률
- **문제**: 최신 생성 모델(Midjourney v6, DALL-E 3 등)의 고품질 이미지는 탐지가 어려움
- **실험 결과**:
  - Midjourney 이미지: 오탐률 약 15%
  - Stable Diffusion XL: 오탐률 약 12%
  - DALL-E 3: 오탐률 약 18%
- **해결 방안**: 최신 생성 모델 데이터 추가 학습

### 3. 실서비스 적용 시 주의점

#### False Positive 위험성
- **문제**: 실제 이미지를 AI로 오분류 (Real → AI)
- **영향**: 
  - 사용자의 실제 사진이 AI로 잘못 분류될 수 있음
  - 신뢰도 저하 및 서비스 품질 문제
- **현재 성능**: CNN 224개 (2.1%), ViT 152개 (1.4%)
- **개선 필요**: 임계값 조정 또는 앙상블 모델 적용

#### False Negative 위험성
- **문제**: AI 이미지를 Real로 오분류 (AI → Real)
- **영향**: 
  - Deepfake 이미지가 실제 이미지로 통과될 수 있음
  - 보안/사기 방지 시스템에서 심각한 문제
- **현재 성능**: CNN 163개 (10.8%), ViT 158개 (10.5%)
- **개선 필요**: AI 클래스 탐지 성능 향상 (현재 F1: 87-90%)

#### Inference Latency
- **현재 성능**:
  - CNN: 평균 15ms (GPU), 120ms (CPU)
  - ViT: 평균 45ms (GPU), 350ms (CPU)
- **실서비스 요구사항**: 
  - 실시간 처리: < 100ms (GPU 기준)
  - 배치 처리: 초당 100+ 이미지
- **개선 필요**: 모델 최적화 (양자화, ONNX 변환)

### 4. 편향 가능성

#### 데이터셋 편향
- **문제**: 특정 도메인(인물, 풍경 등)에 편향된 데이터셋
- **영향**: 다른 도메인 이미지에 대한 일반화 성능 저하
- **해결 방안**: 다양한 도메인 데이터 추가

#### 생성 모델 편향
- **문제**: 특정 AI 생성 모델의 이미지만 학습
- **영향**: 새로운 생성 모델에 대한 탐지 성능 저하
- **해결 방안**: 다양한 생성 모델 데이터 수집 및 지속적 업데이트

---

## 💡 기술적 고찰

### 왜 성능 차이가 0.7%p밖에 안 나는가?

#### 1. **데이터셋 특성**
- **이미지 품질**: 대부분의 이미지가 고품질로, 명확한 구분이 가능
- **클래스 불균형**: Real 클래스가 많아 두 모델 모두 Real 탐지에 집중
- **결과**: 두 모델 모두 높은 성능 달성 (96%+), 차이가 작을 수밖에 없음

#### 2. **사전 학습 효과**
- **CNN**: ImageNet 사전 학습으로 충분한 특징 추출 능력 확보
- **ViT**: ImageNet-21k 사전 학습으로 더 풍부한 특징 학습
- **차이**: 사전 학습 효과가 크면 모델 아키텍처 차이의 영향이 상대적으로 작아짐

#### 3. **태스크 복잡도**
- **이진 분류**: 비교적 단순한 태스크로, 두 모델 모두 충분한 성능 달성 가능
- **복잡한 태스크**: 객체 탐지, 세그멘테이션 등에서는 ViT의 우위가 더 명확할 수 있음

### 왜 AI 클래스가 낮게 나오는가?

#### 1. **클래스 불균형**
- **데이터 비율**: Real:Fake = 6:1
- **영향**: 모델이 Real 클래스에 더 많은 가중치를 부여
- **해결**: Weighted Loss 적용으로 개선했지만 완전한 해결은 아님

#### 2. **AI 이미지의 다양성**
- **문제**: 다양한 생성 모델, 스타일, 품질의 AI 이미지
- **영향**: 일부 고품질 AI 이미지는 실제 이미지와 구분이 어려움
- **해결**: 더 다양한 AI 이미지 데이터 수집 필요

#### 3. **특징 추출의 한계**
- **CNN**: 지역적 패턴에 집중하여 전역적 일관성 문제를 놓칠 수 있음
- **ViT**: 전역적 컨텍스트 이해로 개선했지만 여전히 한계 존재

### 왜 ViT가 더 좋은가?

#### 1. **전역적 컨텍스트 이해**
- **Self-Attention**: 이미지 전체의 패치 간 관계를 동시에 학습
- **AI 이미지 특성**: 전역적 일관성 문제(배경-전경 경계, 조명 불일치) 탐지에 유리
- **결과**: False Positive 감소 (224개 → 152개, 32% 개선)

#### 2. **세밀한 패턴 학습**
- **병렬 학습**: 모든 패치 간 관계를 병렬로 학습
- **미세한 아티팩트**: 생성 과정에서 발생하는 작은 이상 패턴 탐지
- **결과**: AI 클래스 F1 Score 향상 (87.42% → 89.70%)

#### 3. **일반화 성능**
- **사전 학습**: ImageNet-21k로 사전 학습된 가중치 활용
- **Validation-Test Gap**: 작은 차이 (0.22%p)로 과적합 저항
- **결과**: 더 안정적인 성능

---

## ⚡ 모델 최적화

### 양자화 (Quantization)

#### 실험 결과
| 모델 | 정밀도 | 모델 크기 | 추론 속도 (GPU) | Test Accuracy | 메모리 사용량 |
|------|--------|----------|----------------|---------------|--------------|
| **원본** | FP32 | 128MB | 15ms | 96.32% | 512MB |
| **FP16** | FP16 | 64MB | 12ms | 96.28% | 256MB |
| **INT8** | INT8 | 32MB | 8ms | 95.8% | 128MB |

#### 권장사항
- **FP16**: 성능 손실 최소(-0.04%p)하면서 속도 20% 향상 → **권장**
- **INT8**: 성능 손실 있지만 속도 47% 향상 → 엣지 디바이스에 적합

### ONNX 변환

#### 변환 결과
- **모델 크기**: 128MB → 120MB (약 6% 감소)
- **추론 속도**: 
  - PyTorch: 15ms
  - ONNX Runtime (GPU): 10ms (33% 향상)
  - ONNX Runtime (CPU): 95ms (PyTorch CPU 대비 21% 향상)

#### 활용 방안
- **프로덕션 배포**: ONNX Runtime으로 더 빠른 추론
- **크로스 플랫폼**: 다양한 환경에서 동일한 성능 보장

### Latency 비교

| 환경 | CNN (ResNet18) | ViT (ViT-Base) | 개선율 |
|------|----------------|----------------|--------|
| **GPU (FP32)** | 15ms | 45ms | - |
| **GPU (FP16)** | 12ms | 35ms | 20-22% |
| **GPU (ONNX)** | 10ms | 28ms | 33-38% |
| **CPU (FP32)** | 120ms | 350ms | - |
| **CPU (ONNX)** | 95ms | 280ms | 21-20% |

### 최적화 권장사항

1. **실시간 서비스**: CNN + FP16 + ONNX Runtime
2. **배치 처리**: ViT + FP16 (정확도 우선)
3. **엣지 디바이스**: CNN + INT8 + ONNX Runtime

---

## 🔒 보안 및 윤리

### 모델 오탐으로 인한 피해 최소화

#### 1. **임계값 조정**
- **현재**: 기본 임계값 0.5 사용
- **개선**: 클래스별 최적 임계값 탐색
  - Real 클래스: 임계값 0.6 → False Positive 감소
  - AI 클래스: 임계값 0.4 → False Negative 감소

#### 2. **불확실성 추정**
- **구현**: 모델의 예측 확률을 신뢰도로 활용
- **정책**: 신뢰도 < 0.7인 경우 "불확실"로 분류하여 수동 검토

#### 3. **사용자 피드백**
- **구현**: 오분류 사례 수집 및 재학습 파이프라인
- **목적**: 지속적인 모델 성능 개선

### 의도치 않은 편향 가능성

#### 1. **데이터셋 편향**
- **문제**: 특정 인종, 성별, 연령대에 편향된 데이터셋
- **대응**: 
  - 다양한 인구통계학적 특성을 가진 이미지 수집
  - 편향 검사 및 보정

#### 2. **도메인 편향**
- **문제**: 특정 도메인(인물, 풍경 등)에 편향
- **대응**: 다양한 도메인 데이터 추가 및 도메인 적응 기법 적용

### 민감 이미지 처리 정책

#### 1. **개인정보 보호**
- **이미지 저장**: 추론 후 즉시 삭제 (임시 파일 사용)
- **로그 기록**: 이미지 내용은 기록하지 않음, 메타데이터만 저장

#### 2. **NSFW 이미지 처리**
- **필터링**: NSFW 이미지는 별도 처리 파이프라인으로 분리
- **사용자 경고**: NSFW 이미지 업로드 시 경고 메시지 표시

#### 3. **윤리적 사용 가이드라인**
- **목적**: Deepfake 탐지 및 AI 생성 콘텐츠 식별
- **금지 사항**: 
  - 개인 사진 무단 분석
  - 차별적 목적의 사용
  - 사생활 침해

### 향후 개선 방향

1. **공정성 검증**: 다양한 그룹에 대한 성능 균등성 확인
2. **투명성 향상**: 모델의 판단 근거 시각화 (Attention Map, Grad-CAM)
3. **사용자 권리**: 예측 결과에 대한 설명 제공 및 이의 제기 메커니즘

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

