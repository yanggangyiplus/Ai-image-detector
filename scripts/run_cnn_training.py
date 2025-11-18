#!/usr/bin/env python3
"""
CNN 모델 학습 스크립트
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
from models.model_utils import create_cnn_model, print_model_info
from data.dataset import ImageDataset
from data.preprocess import get_train_transforms, get_val_transforms
from training.loss_functions import get_loss_function
from training.optimizer import create_optimizer, create_scheduler
from training.train import train_model, EarlyStopping
from training.metrics import calculate_all_metrics, print_metrics, plot_confusion_matrix
from training.evaluator import evaluate_model
from utils.seed import set_seed

print("=" * 70)
print("CNN 모델 학습 시작")
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

# 결과 디렉토리 생성
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 설정
CONFIG = {
    'model_name': 'resnet18',
    'num_classes': 2,
    'batch_size': 32,
    'num_epochs': 50,
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'image_size': 224,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'class_names': ['real', 'fake'],
    'class_map': {'real': '0', 'fake': '1'}
}

print(f"\n설정:")
print(f"  디바이스: {CONFIG['device']}")
print(f"  모델: {CONFIG['model_name']}")
print(f"  배치 크기: {CONFIG['batch_size']}")
print(f"  에포크 수: {CONFIG['num_epochs']}")
print(f"  학습률: {CONFIG['learning_rate']}")

# 데이터셋 및 데이터로더 생성
print(f"\n데이터셋 로드 중...")
train_transform = get_train_transforms(CONFIG['image_size'])
val_transform = get_val_transforms(CONFIG['image_size'])
test_transform = get_val_transforms(CONFIG['image_size'])

train_dataset = ImageDataset(TRAIN_DIR, CONFIG['class_map'], transform=train_transform)
val_dataset = ImageDataset(VAL_DIR, CONFIG['class_map'], transform=val_transform)
test_dataset = ImageDataset(TEST_DIR, CONFIG['class_map'], transform=test_transform)

# macOS에서는 num_workers=0으로 설정 (multiprocessing 이슈 방지)
train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)

print(f"  훈련 데이터: {len(train_dataset):,}개")
print(f"  검증 데이터: {len(val_dataset):,}개")
print(f"  테스트 데이터: {len(test_dataset):,}개")

# 모델 생성
print(f"\n모델 생성 중...")
model = create_cnn_model(
    model_name=CONFIG['model_name'],
    num_classes=CONFIG['num_classes'],
    pretrained=True
)
model = model.to(CONFIG['device'])
print_model_info(model)

# Loss, Optimizer, Scheduler 설정
print(f"\n학습 설정 중...")
criterion = get_loss_function('cross_entropy')
optimizer = create_optimizer(
    model,
    'adamw',
    learning_rate=CONFIG['learning_rate'],
    weight_decay=CONFIG['weight_decay']
)
scheduler = create_scheduler(
    optimizer,
    'cosine_annealing',
    num_epochs=CONFIG['num_epochs']
)
early_stopping = EarlyStopping(patience=10, min_delta=0.001)

# 학습 실행
print(f"\n{'='*70}")
print(f"학습 시작!")
print(f"{'='*70}\n")
model_name = f"CNN_{CONFIG['model_name']}"

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
    scheduler_name='cosine_annealing'
)

print(f"\n{'='*70}")
print(f"학습 완료! 테스트 세트 평가 시작...")
print(f"{'='*70}\n")

# Best 모델 로드 및 테스트 평가
best_model_path = CHECKPOINT_DIR / f'{model_name}_best.pth'
checkpoint = torch.load(best_model_path, map_location=CONFIG['device'], weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"✅ Best 모델 로드 완료 (Epoch {checkpoint['epoch']})")

test_results = evaluate_model(
    model,
    test_loader,
    CONFIG['device'],
    class_names=CONFIG['class_names']
)

print_metrics(test_results['metrics'], class_names=CONFIG['class_names'])

# 결과 저장
test_metrics = {
    'accuracy': test_results['metrics']['accuracy'],
    'precision': test_results['metrics']['precision'],
    'recall': test_results['metrics']['recall'],
    'f1': test_results['metrics']['f1'],
    'confusion_matrix': test_results['metrics']['confusion_matrix'].tolist()
}

with open(RESULTS_DIR / f'{model_name}_test_metrics.json', 'w', encoding='utf-8') as f:
    json.dump(test_metrics, f, indent=2, ensure_ascii=False)

# Loss Curve 저장
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(history['train_loss'], label='Train Loss', color='blue')
axes[0].plot(history['val_loss'], label='Val Loss', color='red')
axes[0].set_title('Loss Curve', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history['train_accuracy'], label='Train Accuracy', color='blue')
axes[1].plot(history['val_accuracy'], label='Val Accuracy', color='red')
axes[1].set_title('Accuracy Curve', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS_DIR / f'{model_name}_curves.png', dpi=300, bbox_inches='tight')
plt.close()

# Confusion Matrix 저장
cm = test_results['metrics']['confusion_matrix']
plot_confusion_matrix(cm, CONFIG['class_names'], save_path=RESULTS_DIR / f'{model_name}_confusion_matrix.png')

print(f"\n{'='*70}")
print(f"✅ 모든 작업 완료!")
print(f"{'='*70}")
print(f"\n저장된 파일:")
print(f"  - 체크포인트: {CHECKPOINT_DIR / f'{model_name}_best.pth'}")
print(f"  - Loss Curve: {RESULTS_DIR / f'{model_name}_curves.png'}")
print(f"  - Confusion Matrix: {RESULTS_DIR / f'{model_name}_confusion_matrix.png'}")
print(f"  - 메트릭: {RESULTS_DIR / f'{model_name}_test_metrics.json'}")
print(f"{'='*70}")

