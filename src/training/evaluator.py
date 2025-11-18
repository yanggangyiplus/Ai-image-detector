"""
검증 및 평가 함수 모듈
"""
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from .metrics import calculate_metrics, calculate_class_wise_metrics


def evaluate_model(model, dataloader, device, class_names=None):
    """
    모델 평가 함수
    
    Args:
        model: 평가할 모델
        dataloader: 데이터 로더
        device: 디바이스
        class_names: 클래스 이름 리스트
        
    Returns:
        results: 평가 결과 딕셔너리
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 메트릭 계산
    metrics = calculate_metrics(all_labels, all_preds)
    
    results = {
        'metrics': metrics,
        'predictions': all_preds,
        'labels': all_labels
    }
    
    # 클래스별 메트릭 계산
    if class_names:
        class_metrics = calculate_class_wise_metrics(all_labels, all_preds, class_names)
        results['class_metrics'] = class_metrics
    
    return results

