"""
학습 곡선 그래프 생성 스크립트
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 경로 설정
PROJECT_ROOT = Path(__file__).parent.parent
LOG_DIR = PROJECT_ROOT / "experiments" / "logs"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def load_history(model_name):
    """학습 히스토리 로드"""
    history_path = LOG_DIR / f"{model_name}_history.json"
    if history_path.exists():
        with open(history_path, 'r') as f:
            return json.load(f)
    return None

def plot_training_curves(model_name, history, save_path):
    """학습 곡선 그리기"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss Curve
    axes[0].plot(epochs, history['train_loss'], label='Train Loss', color='#3498db', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], label='Val Loss', color='#e74c3c', linewidth=2)
    axes[0].set_title(f'{model_name} - Loss Curve', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy Curve
    axes[1].plot(epochs, history['train_accuracy'], label='Train Accuracy', color='#3498db', linewidth=2)
    axes[1].plot(epochs, history['val_accuracy'], label='Val Accuracy', color='#e74c3c', linewidth=2)
    axes[1].set_title(f'{model_name} - Accuracy Curve', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 그래프 저장: {save_path}")
    plt.close()

def plot_comparison_curves(cnn_history, vit_history, save_path):
    """모델 비교 곡선 그리기"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    cnn_epochs = range(1, len(cnn_history['train_loss']) + 1)
    vit_epochs = range(1, len(vit_history['train_loss']) + 1)
    
    # Loss Curve 비교
    axes[0].plot(cnn_epochs, cnn_history['train_loss'], label='CNN Train', color='#3498db', linestyle='--', alpha=0.7)
    axes[0].plot(cnn_epochs, cnn_history['val_loss'], label='CNN Val', color='#3498db', linewidth=2)
    axes[0].plot(vit_epochs, vit_history['train_loss'], label='ViT Train', color='#e74c3c', linestyle='--', alpha=0.7)
    axes[0].plot(vit_epochs, vit_history['val_loss'], label='ViT Val', color='#e74c3c', linewidth=2)
    axes[0].set_title('Loss Curve Comparison', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy Curve 비교
    axes[1].plot(cnn_epochs, cnn_history['train_accuracy'], label='CNN Train', color='#3498db', linestyle='--', alpha=0.7)
    axes[1].plot(cnn_epochs, cnn_history['val_accuracy'], label='CNN Val', color='#3498db', linewidth=2)
    axes[1].plot(vit_epochs, vit_history['train_accuracy'], label='ViT Train', color='#e74c3c', linestyle='--', alpha=0.7)
    axes[1].plot(vit_epochs, vit_history['val_accuracy'], label='ViT Val', color='#e74c3c', linewidth=2)
    axes[1].set_title('Accuracy Curve Comparison', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 비교 그래프 저장: {save_path}")
    plt.close()

def main():
    """메인 함수"""
    print("=" * 70)
    print("학습 곡선 그래프 생성")
    print("=" * 70)
    
    # CNN 히스토리 로드
    cnn_history = load_history("CNN_resnet18")
    if cnn_history:
        plot_training_curves("CNN (ResNet18)", cnn_history, 
                           RESULTS_DIR / "CNN_resnet18_training_curves.png")
    
    # ViT 히스토리 로드
    vit_history = load_history("ViT_vit_base")
    if vit_history:
        plot_training_curves("ViT (ViT-Base)", vit_history,
                           RESULTS_DIR / "ViT_vit_base_training_curves.png")
    
    # 비교 그래프 생성
    if cnn_history and vit_history:
        plot_comparison_curves(cnn_history, vit_history,
                              RESULTS_DIR / "model_comparison_curves.png")
    
    print("\n✅ 모든 그래프 생성 완료!")

if __name__ == "__main__":
    main()

