# 프로덕션 배포 가이드

## Auto-scaling (GPU Inference 서버 수평 확장)

### Kubernetes Deployment 예시
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-image-detector-api
spec:
  replicas: 3  # 최소 3개 Pod 유지
  template:
    spec:
      containers:
      - name: api
        image: ai-image-detector:latest
        resources:
          requests:
            nvidia.com/gpu: 1
          limits:
            nvidia.com/gpu: 1
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ai-image-detector-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ai-image-detector-api
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

## Model Registry (MLflow, HuggingFace Hub)

### 모델 버전 관리
```
Model Registry 구조:
├── v1.0_cnn_resnet18_fp32
│   ├── model.pth
│   ├── config.yaml
│   └── metrics.json
├── v1.1_cnn_resnet18_fp16
│   ├── model.pth
│   ├── config.yaml
│   └── metrics.json
└── v2.0_vit_base_fp16
    ├── model.pth
    ├── config.yaml
    └── metrics.json
```

### MLflow 통합
```python
import mlflow
import mlflow.pytorch

# 모델 등록
mlflow.set_tracking_uri("http://mlflow-server:5000")
mlflow.set_experiment("ai-image-detector")

with mlflow.start_run():
    mlflow.log_params({
        "model": "resnet18",
        "batch_size": 32,
        "learning_rate": 1e-4
    })
    mlflow.log_metrics({
        "test_accuracy": 0.9632,
        "test_f1": 0.9635
    })
    mlflow.pytorch.log_model(model, "model")
```

### HuggingFace Hub 통합
```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_file(
    path_or_fileobj="experiments/checkpoints/CNN_resnet18_best.pth",
    path_in_repo="models/cnn_resnet18.pth",
    repo_id="yanggangyi/ai-image-detector",
    repo_type="model"
)
```

## CI/CD (GitHub Actions → Docker → HuggingFace 자동 빌드)

### GitHub Actions 워크플로우
```yaml
# .github/workflows/deploy.yml
name: Deploy to HuggingFace Spaces

on:
  push:
    branches: [main]
    paths:
      - 'app/**'
      - 'src/**'
      - 'deployment/huggingface/**'

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Build Docker image
        run: |
          cd deployment/huggingface
          docker build -t ai-image-detector .
      
      - name: Push to HuggingFace
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
        run: |
          docker login -u ${{ secrets.HF_USERNAME }} -p $HF_TOKEN
          docker push ${{ secrets.HF_USERNAME }}/ai-image-detector:latest
      
      - name: Deploy to Spaces
        run: |
          curl -X POST https://huggingface.co/api/spaces/${{ secrets.HF_USERNAME }}/Ai-image-detector/restart \
            -H "Authorization: Bearer $HF_TOKEN"
```

## 모니터링 (Prometheus + Grafana)

### 메트릭 수집
```python
from prometheus_client import Counter, Histogram, Gauge

# 메트릭 정의
inference_requests = Counter('inference_requests_total', 'Total inference requests')
inference_latency = Histogram('inference_latency_seconds', 'Inference latency')
model_accuracy = Gauge('model_accuracy', 'Current model accuracy')
gpu_utilization = Gauge('gpu_utilization_percent', 'GPU utilization')

# 메트릭 기록
@inference_latency.time()
def predict(image):
    inference_requests.inc()
    result = model(image)
    gpu_utilization.set(get_gpu_utilization())
    return result
```

### Grafana 대시보드
- **실시간 메트릭**: 요청 수, 응답 시간, 에러율
- **모델 성능**: 정확도, F1 Score, Confusion Matrix
- **리소스 사용량**: GPU/CPU 사용률, 메모리 사용량
- **알림**: 성능 저하, 에러율 증가 시 알림

## 로깅 및 에러 추적

### 구조화된 로깅
```python
import logging
import json

logger = logging.getLogger(__name__)

def log_inference(image_id, prediction, confidence, latency):
    logger.info(json.dumps({
        "event": "inference",
        "image_id": image_id,
        "prediction": prediction,
        "confidence": confidence,
        "latency_ms": latency,
        "timestamp": datetime.now().isoformat()
    }))
```

### 에러 추적 (Sentry)
```python
import sentry_sdk

sentry_sdk.init(
    dsn="your-sentry-dsn",
    traces_sample_rate=1.0,
)

try:
    result = predict(image)
except Exception as e:
    sentry_sdk.capture_exception(e)
    raise
```

## 성능 최적화

### 캐싱 전략
```python
from functools import lru_cache
import redis

redis_client = redis.Redis(host='localhost', port=6379)

def predict_with_cache(image_hash):
    # Redis 캐시 확인
    cached_result = redis_client.get(f"prediction:{image_hash}")
    if cached_result:
        return json.loads(cached_result)
    
    # 모델 추론
    result = model.predict(image)
    
    # 캐시 저장 (TTL: 1시간)
    redis_client.setex(
        f"prediction:{image_hash}",
        3600,
        json.dumps(result)
    )
    return result
```

### 배치 처리 최적화
```python
async def batch_predict(images: List[Image]):
    # 배치 크기 최적화
    batch_size = 32
    results = []
    
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        batch_results = model.predict_batch(batch)
        results.extend(batch_results)
    
    return results
```


