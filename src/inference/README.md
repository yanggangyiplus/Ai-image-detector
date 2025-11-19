# Inference 모듈 사용 가이드

## 개요

이 모듈은 학습된 모델을 사용하여 이미지 추론을 수행하는 기능을 제공합니다.

## 주요 기능

### 1. 단일 이미지 추론 (`inference.py`)

단일 이미지에 대한 예측을 수행합니다.

#### 주요 함수

- `load_image(image_path, transform=None, image_size=224)`: 이미지 로드 및 전처리
- `predict_single_image(model, image_path, device='cpu', class_names=None)`: 단일 이미지 예측
- `load_model_for_inference(checkpoint_path, model_type='cnn', model_name='resnet18', ...)`: 모델 로드
- `print_prediction_result(result, verbose=True)`: 결과 출력
- `save_prediction_result(result, save_path)`: 결과 JSON 저장

#### 사용 예제

```python
from src.inference.inference import (
    load_model_for_inference,
    predict_single_image,
    print_prediction_result
)
import torch

# 디바이스 설정
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 모델 로드
model, checkpoint = load_model_for_inference(
    checkpoint_path='experiments/checkpoints/CNN_resnet18_best.pth',
    model_type='cnn',
    model_name='resnet18',
    num_classes=2,
    device=device
)

# 단일 이미지 추론
result = predict_single_image(
    model=model,
    image_path='path/to/image.jpg',
    device=device,
    class_names=['Real', 'AI']
)

# 결과 출력
print_prediction_result(result, verbose=True)
```

#### 결과 형식

```python
{
    'image_path': 'path/to/image.jpg',
    'predicted_class': 'Real',  # 또는 'AI'
    'predicted_class_idx': 0,  # 0: Real, 1: AI
    'confidence': 0.9876,  # 예측 신뢰도 (0-1)
    'probabilities': {
        'Real': 0.9876,
        'AI': 0.0124
    },
    'is_ai': False  # AI 이미지 여부
}
```

### 2. 배치 이미지 추론 (`batch_inference.py`)

여러 이미지를 한 번에 처리합니다.

#### 주요 함수

- `batch_predict(model, image_paths, device='cpu', batch_size=32, ...)`: 배치 예측
- `save_batch_results(results, save_path, format='json')`: 결과 저장 (JSON/CSV)
- `print_batch_summary(results, class_names=None)`: 요약 정보 출력

#### 사용 예제

```python
from src.inference.inference import load_model_for_inference
from src.inference.batch_inference import (
    batch_predict,
    save_batch_results,
    print_batch_summary
)
import torch

# 모델 로드
model, checkpoint = load_model_for_inference(
    checkpoint_path='experiments/checkpoints/ViT_vit_base_best.pth',
    model_type='vit',
    model_name='vit_base',
    num_classes=2,
    device='cpu'
)

# 배치 추론
image_paths = [
    'path/to/image1.jpg',
    'path/to/image2.jpg',
    'path/to/image3.jpg'
]

results = batch_predict(
    model=model,
    image_paths=image_paths,  # 또는 디렉토리 경로
    device='cpu',
    batch_size=32,
    class_names=['Real', 'AI']
)

# 요약 출력
print_batch_summary(results, class_names=['Real', 'AI'])

# 결과 저장
save_batch_results(results, 'results.json', format='json')
save_batch_results(results, 'results.csv', format='csv')
```

#### 디렉토리 처리

디렉토리 경로를 전달하면 자동으로 모든 이미지 파일을 찾아 처리합니다:

```python
results = batch_predict(
    model=model,
    image_paths='path/to/image_directory/',  # 디렉토리 경로
    device='cpu',
    batch_size=32,
    class_names=['Real', 'AI']
)
```

## 명령줄 사용법

### 단일 이미지 추론

```bash
# 기본 사용 (CNN 모델)
python examples/single_image_inference.py path/to/image.jpg

# ViT 모델 사용
python examples/single_image_inference.py path/to/image.jpg --model ViT

# 결과 저장
python examples/single_image_inference.py path/to/image.jpg --save result.json

# 상세 정보 없이 출력
python examples/single_image_inference.py path/to/image.jpg --no-verbose
```

### 배치 이미지 추론

```bash
# 디렉토리 내 모든 이미지 처리
python examples/batch_inference.py path/to/image_directory/

# 특정 모델 사용
python examples/batch_inference.py path/to/image_directory/ --model ViT

# 배치 크기 조정
python examples/batch_inference.py path/to/image_directory/ --batch_size 16

# 결과를 CSV로 저장
python examples/batch_inference.py path/to/image_directory/ --output results.csv --format csv

# 요약 정보 없이 출력
python examples/batch_inference.py path/to/image_directory/ --no-summary
```

## 출력 예시

### 단일 이미지 추론 결과

```
============================================================
📸 이미지 추론 결과
============================================================
이미지 경로: path/to/image.jpg

예측 결과:
  클래스: Real
  신뢰도: 0.9876 (98.76%)
  판단: 📷 실제 이미지

모든 클래스 확률:
  Real           : 0.9876 ( 98.76%) ██████████████████████████████
  AI             : 0.0124 (  1.24%) ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
============================================================
```

### 배치 추론 요약

```
============================================================
배치 추론 결과 요약
============================================================
총 처리된 이미지: 100개

클래스별 분포:
  Real: 85개 (85.00%)
  AI: 15개 (15.00%)

평균 신뢰도: 0.9234 (92.34%)

신뢰도 분포:
  높음 (≥90%): 78개 (78.00%)
  중간 (70-90%): 18개 (18.00%)
  낮음 (<70%): 4개 (4.00%)
============================================================
```

## 주의사항

1. **디바이스 설정**: GPU가 있으면 자동으로 사용하지만, 명시적으로 지정할 수도 있습니다.
2. **배치 크기**: GPU 메모리에 따라 조정하세요. ViT는 더 많은 메모리를 사용합니다.
3. **이미지 형식**: 지원 형식: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.tif`
4. **macOS 호환성**: `num_workers=0`으로 설정하는 것을 권장합니다.

## 에러 처리

- 존재하지 않는 이미지 파일은 자동으로 건너뜁니다.
- 이미지 로드 실패 시 명확한 에러 메시지를 제공합니다.
- 체크포인트 파일이 없으면 에러를 발생시킵니다.



