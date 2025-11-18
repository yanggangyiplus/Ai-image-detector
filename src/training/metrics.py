"""
평가 메트릭 정의 모듈
Accuracy, Precision, Recall, F1, Confusion Matrix 등 제공
"""
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
from typing import Dict, List, Tuple


def calculate_accuracy(outputs, labels):
    """
    정확도 계산
    
    Args:
        outputs: 모델 출력
        labels: 실제 레이블
        
    Returns:
        accuracy: 정확도
    """
    _, predicted = torch.max(outputs.data, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    return 100 * correct / total


def calculate_metrics(y_true, y_pred, average='weighted'):
    """
    다양한 메트릭 계산
    
    Args:
        y_true: 실제 레이블
        y_pred: 예측 레이블
        average: 평균 방법 ('micro', 'macro', 'weighted')
        
    Returns:
        metrics: 메트릭 딕셔너리
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average),
        'recall': recall_score(y_true, y_pred, average=average),
        'f1': f1_score(y_true, y_pred, average=average),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }
    return metrics


def calculate_class_wise_metrics(y_true, y_pred, class_names):
    """
    클래스별 메트릭 계산
    
    Args:
        y_true: 실제 레이블
        y_pred: 예측 레이블
        class_names: 클래스 이름 리스트
        
    Returns:
        class_metrics: 클래스별 메트릭 딕셔너리
    """
    cm = confusion_matrix(y_true, y_pred)
    class_metrics = {}
    
    for i, class_name in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        class_metrics[class_name] = {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    return class_metrics


def calculate_all_metrics(outputs, labels, class_names=None):
    """
    모든 메트릭을 한 번에 계산
    
    Args:
        outputs: 모델 출력 (logits)
        labels: 실제 레이블
        class_names: 클래스 이름 리스트 (선택)
        
    Returns:
        metrics: 모든 메트릭을 포함한 딕셔너리
    """
    # 예측값 계산
    _, predicted = torch.max(outputs.data, 1)
    y_pred = predicted.cpu().numpy()
    y_true = labels.cpu().numpy()
    
    # 기본 메트릭 계산
    metrics = calculate_metrics(y_true, y_pred, average='weighted')
    
    # 클래스별 메트릭 추가
    if class_names is not None:
        class_metrics = calculate_class_wise_metrics(y_true, y_pred, class_names)
        metrics['class_wise'] = class_metrics
    
    return metrics


def print_metrics(metrics, class_names=None):
    """
    메트릭을 보기 좋게 출력
    
    Args:
        metrics: 메트릭 딕셔너리
        class_names: 클래스 이름 리스트
    """
    print("\n" + "=" * 60)
    print("평가 메트릭")
    print("=" * 60)
    print(f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    
    if 'class_wise' in metrics and class_names:
        print("\n클래스별 메트릭:")
        for class_name in class_names:
            if class_name in metrics['class_wise']:
                class_met = metrics['class_wise'][class_name]
                print(f"  {class_name}:")
                print(f"    Precision: {class_met['precision']:.4f}")
                print(f"    Recall:    {class_met['recall']:.4f}")
                print(f"    F1:        {class_met['f1']:.4f}")
    
    print("\n혼동 행렬:")
    print(metrics['confusion_matrix'])
    print("=" * 60)


def plot_confusion_matrix(cm, class_names, save_path=None):
    """
    혼동 행렬 시각화
    
    Args:
        cm: 혼동 행렬
        class_names: 클래스 이름 리스트
        save_path: 저장 경로 (선택)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    plt.figure(figsize=(8, 6))
    
    # matplotlib만 사용하여 heatmap 생성
    im = plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im)
    
    # 클래스 이름 설정
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    
    # 숫자 표시
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"혼동 행렬 저장: {save_path}")
    
    plt.show()

