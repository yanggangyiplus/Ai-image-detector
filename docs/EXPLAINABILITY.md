# 모델 Explainability 상세 가이드

## 모델 해석 가능성 (Interpretability)

AI 이미지 탐지 모델의 판단 근거를 시각화하여 **"왜 이 이미지가 AI 생성 이미지로 판단되었는가"**를 설명할 수 있습니다.

## CNN - Grad-CAM (Gradient-weighted Class Activation Mapping)

### 원리
CNN의 마지막 컨볼루션 레이어에서 각 픽셀이 예측에 기여한 정도를 시각화

### 활용
- AI 이미지로 판단된 **핵심 영역** 강조
- 모델이 집중하는 **패턴** 파악
- 오분류 사례의 **원인 분석**

### 예시 시나리오
```
AI 이미지 탐지 시 Grad-CAM이 강조하는 영역:
- 부자연스러운 배경-전경 경계
- 비정상적인 조명 패턴
- 반복적인 텍스처 패턴
- 얼굴/손의 비정상적인 구조
```

## ViT - Attention Map Visualization

### 원리
Self-Attention 메커니즘에서 각 패치가 다른 패치에 부여하는 어텐션 가중치를 시각화

### 활용
- 이미지 전체의 **패치 간 관계** 파악
- 전역적 **일관성 문제** 탐지 영역 확인
- 모델의 **의사결정 과정** 이해

### 예시 시나리오
```
ViT Attention Map이 보여주는 것:
- 배경과 전경의 불일치 (높은 어텐션)
- 조명 방향의 불일치 (패치 간 어텐션 패턴)
- 미세한 아티팩트 영역 (집중된 어텐션)
```

## 구현 방법

### Grad-CAM (CNN)
```python
from src.models.cnn import ResNetClassifier
from src.inference.explainability import generate_gradcam

model = ResNetClassifier(num_classes=2)
# ... 모델 로드 ...

# Grad-CAM 생성
gradcam = generate_gradcam(
    model=model,
    image=image_tensor,
    target_layer='layer4',  # 마지막 컨볼루션 레이어
    class_idx=1  # AI 클래스
)
```

### Attention Map (ViT)
```python
from transformers import ViTModel
from src.inference.explainability import visualize_attention

model = ViTModel.from_pretrained('google/vit-base-patch16-224')
# ... 모델 로드 ...

# Attention Map 시각화
attention_map = visualize_attention(
    model=model,
    image=image_tensor,
    head_idx=0,  # 첫 번째 어텐션 헤드
    layer_idx=-1  # 마지막 레이어
)
```

## 향후 개선 방향

1. **실시간 Explainability**: 웹 데모에 Grad-CAM/Attention Map 통합
2. **대화형 시각화**: 사용자가 클릭한 영역의 기여도 확인
3. **오분류 분석**: False Positive/Negative 사례의 시각적 설명
4. **SHAP Values**: 특징 기여도 정량화

