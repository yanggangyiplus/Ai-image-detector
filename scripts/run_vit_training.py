#!/usr/bin/env python3
"""
ViT 모델 Fine-tuning 스크립트
학습 진행 상황을 실시간으로 확인할 수 있습니다.
"""
import sys
import os
from pathlib import Path

# 프로젝트 루트를 경로에 추가
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # GUI 없이 실행
import matplotlib.pyplot as plt
import numpy as np
import json
from tqdm import tqdm

# 프로젝트 모듈 import
from models.model_utils import create_vit_model, print_model_info
from data.dataset import ImageDataset
from data.preprocess import get_train_transforms, get_val_transforms, get_test_transforms
from training.loss_functions import get_loss_function
from training.optimizer import create_optimizer, create_scheduler
from training.train import train_model, EarlyStopping
from training.metrics import calculate_metrics, calculate_class_wise_metrics, print_metrics, plot_confusion_matrix
from utils.seed import set_seed

print("=" * 70)
print("ViT 모델 Fine-tuning 시작")
print("=" * 70)

# 시드 고정
set_seed(42)

# 경로 설정
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"
CHECKPOINT_DIR = PROJECT_ROOT / "experiments/checkpoints"
LOG_DIR = PROJECT_ROOT / "experiments/logs"
RESULTS_DIR = PROJECT_ROOT / "experiments/results"
CLASS_MAP_PATH = str(DATA_DIR / "class_map.json")

# 결과 디렉토리 생성
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 설정
CONFIG = {
    'model_name': 'vit_base',  # 'vit_base', 'vit_for_classification'
    'num_classes': 2,
    'batch_size': 16,  # ViT는 메모리를 더 많이 사용하므로 작은 배치 크기
    'num_epochs': 30,
    'learning_rate': 1e-5,  # Fine-tuning을 위해 작은 학습률
    'weight_decay': 1e-4,
    'image_size': 224,
    'device': 'cuda' if torch.cuda.is_available() else 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu',
    'class_names': ['Real', 'AI'],
    'freeze_backbone': False,  # True로 설정하면 일부 레이어 고정
    'freeze_layers': None  # 고정할 레이어 수 (None이면 자동)
}

print(f"디바이스: {CONFIG['device']}")
print(f"모델: {CONFIG['model_name']}")
print(f"배치 크기: {CONFIG['batch_size']}")
print(f"에포크 수: {CONFIG['num_epochs']}")
print(f"학습률: {CONFIG['learning_rate']}")
print(f"Fine-tuning 모드: freeze_backbone={CONFIG['freeze_backbone']}\n")

# 데이터셋 및 데이터로더 생성
print("데이터셋 준비 중...")
train_transform = get_train_transforms(CONFIG['image_size'])
val_transform = get_val_transforms(CONFIG['image_size'])
test_transform = get_test_transforms(CONFIG['image_size'])

train_dataset = ImageDataset(
    data_dir=str(TRAIN_DIR),
    class_map=CLASS_MAP_PATH,
    transform=train_transform
)
val_dataset = ImageDataset(
    data_dir=str(VAL_DIR),
    class_map=CLASS_MAP_PATH,
    transform=val_transform
)
test_dataset = ImageDataset(
    data_dir=str(TEST_DIR),
    class_map=CLASS_MAP_PATH,
    transform=test_transform
)

train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)

print(f"✅ 학습 데이터셋: {len(train_dataset)}개 샘플")
print(f"✅ 검증 데이터셋: {len(val_dataset)}개 샘플")
print(f"✅ 테스트 데이터셋: {len(test_dataset)}개 샘플\n")

# ViT 모델 생성
print("ViT 모델 생성 중...")
model = create_vit_model(
    model_name=CONFIG['model_name'],
    num_classes=CONFIG['num_classes'],
    pretrained=True,
    freeze_backbone=CONFIG['freeze_backbone'],
    freeze_layers=CONFIG['freeze_layers']
)
model = model.to(CONFIG['device'])
print_model_info(model)
print()

# Loss 함수
criterion = get_loss_function('cross_entropy')

# Optimizer
optimizer = create_optimizer(
    model,
    'adamw',
    learning_rate=CONFIG['learning_rate'],
    weight_decay=CONFIG['weight_decay']
)

# Scheduler (ViT는 ReduceLROnPlateau가 더 효과적)
scheduler = create_scheduler(
    optimizer,
    'reduce_lr_on_plateau',
    patience=5,
    factor=0.5
)

# Early Stopping
early_stopping = EarlyStopping(patience=10, min_delta=0.001)

print("=" * 70)
print("Fine-tuning 시작")
print("=" * 70)

# Fine-tuning 실행
model_name = f"ViT_{CONFIG['model_name']}"
history = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    num_epochs=CONFIG['num_epochs'],
    device=CONFIG['device'],
    class_names=CONFIG['class_names'],
    save_dir=CHECKPOINT_DIR,
    log_dir=LOG_DIR,
    model_name=model_name,
    early_stopping=early_stopping,
    scheduler_name='reduce_lr_on_plateau'
)

print(f"\n{'='*70}")
print(f"학습 완료! 테스트 세트 평가 시작...")
print(f"{'='*70}\n")

# Best 모델 로드 및 테스트 평가
best_model_path = CHECKPOINT_DIR / f'{model_name}_best.pth'
checkpoint = torch.load(best_model_path, map_location=CONFIG['device'], weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

val_metrics = checkpoint.get('val_metrics', {})
best_epoch = checkpoint.get('epoch', 0)
print(f"✅ Best 모델 로드 완료 (Epoch {best_epoch})")
if val_metrics:
    print(f"   Best Val Loss: {val_metrics.get('loss', 'N/A'):.4f}")
    print(f"   Best Val Accuracy: {val_metrics.get('accuracy', 'N/A'):.4f}\n")

# 테스트 평가
print("테스트 세트 평가 중...")
all_preds = []
all_labels = []

with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(test_loader):
        images = images.to(CONFIG['device'])
        labels = labels.to(CONFIG['device'])
        
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        if (batch_idx + 1) % 20 == 0:
            print(f"  진행 중... {batch_idx + 1}/{len(test_loader)} 배치 완료")

# 메트릭 계산
y_true = np.array(all_labels)
y_pred = np.array(all_preds)
metrics = calculate_metrics(y_true, y_pred, average='weighted')
class_metrics = calculate_class_wise_metrics(y_true, y_pred, class_names=CONFIG['class_names'])

test_results = {
    'accuracy': metrics['accuracy'],
    'precision': metrics['precision'],
    'recall': metrics['recall'],
    'f1': metrics['f1'],
    'confusion_matrix': metrics['confusion_matrix'],
    'class_wise': class_metrics
}

print("\n" + "="*70)
print("테스트 세트 평가 결과")
print("="*70)
print_metrics(test_results, class_names=CONFIG['class_names'])

# Confusion Matrix 저장
Path('experiments/results').mkdir(parents=True, exist_ok=True)
plot_confusion_matrix(metrics['confusion_matrix'], class_names=CONFIG['class_names'], 
                      save_path=f'experiments/results/{model_name}_test_confusion_matrix.png')
print(f"\n✅ Confusion Matrix 저장: experiments/results/{model_name}_test_confusion_matrix.png")

print("\n✅ ViT 모델 Fine-tuning 및 평가 완료!")

