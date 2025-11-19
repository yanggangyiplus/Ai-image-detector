"""
옵티마이저 및 스케줄러 설정 모듈
AdamW 옵티마이저와 Cosine Annealing, ReduceLROnPlateau 스케줄러 제공
"""
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from typing import Optional, Dict, Any


def create_optimizer(model, optimizer_name='adamw', learning_rate=1e-4, 
                    weight_decay=1e-4, **kwargs):
    """
    옵티마이저 생성 함수
    
    Args:
        model: 학습할 모델
        optimizer_name: 옵티마이저 이름 ('adamw', 'adam', 'sgd')
        learning_rate: 학습률
        weight_decay: 가중치 감쇠 (L2 정규화)
        **kwargs: 옵티마이저별 추가 인자
        
    Returns:
        optimizer: 옵티마이저 객체
        
    Example:
        >>> optimizer = create_optimizer(model, 'adamw', learning_rate=1e-4, weight_decay=1e-4)
        >>> optimizer = create_optimizer(model, 'adam', learning_rate=1e-3)
    """
    # 학습 가능한 파라미터만 선택
    params = [p for p in model.parameters() if p.requires_grad]
    
    if optimizer_name.lower() == 'adamw':
        # AdamW: Adam의 개선 버전 (가중치 감쇠 분리)
        betas = kwargs.get('betas', (0.9, 0.999))
        eps = kwargs.get('eps', 1e-8)
        optimizer = optim.AdamW(
            params,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps
        )
    
    elif optimizer_name.lower() == 'adam':
        # Adam 옵티마이저
        betas = kwargs.get('betas', (0.9, 0.999))
        eps = kwargs.get('eps', 1e-8)
        optimizer = optim.Adam(
            params,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps
        )
    
    elif optimizer_name.lower() == 'sgd':
        # SGD 옵티마이저
        momentum = kwargs.get('momentum', 0.9)
        nesterov = kwargs.get('nesterov', False)
        optimizer = optim.SGD(
            params,
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov
        )
    
    else:
        raise ValueError(
            f"지원하지 않는 옵티마이저: {optimizer_name}\n"
            f"지원 옵티마이저: ['adamw', 'adam', 'sgd']"
        )
    
    print(f"{optimizer_name.upper()} 옵티마이저 생성 완료")
    print(f"   학습률: {learning_rate}")
    print(f"   가중치 감쇠: {weight_decay}")
    print(f"   학습 가능한 파라미터 수: {sum(p.numel() for p in params):,}")
    
    return optimizer


def create_scheduler(optimizer, scheduler_name='cosine_annealing', 
                    num_epochs=None, **kwargs):
    """
    학습률 스케줄러 생성 함수
    
    Args:
        optimizer: 옵티마이저 객체
        scheduler_name: 스케줄러 이름 ('cosine_annealing', 'reduce_lr_on_plateau')
        num_epochs: 총 에포크 수 (Cosine Annealing용)
        **kwargs: 스케줄러별 추가 인자
        
    Returns:
        scheduler: 스케줄러 객체
        
    Example:
        >>> scheduler = create_scheduler(optimizer, 'cosine_annealing', num_epochs=50)
        >>> scheduler = create_scheduler(optimizer, 'reduce_lr_on_plateau', 
        ...                              patience=5, factor=0.5)
    """
    if scheduler_name.lower() == 'cosine_annealing':
        # Cosine Annealing: 코사인 함수 형태로 학습률 감소
        if num_epochs is None:
            raise ValueError("Cosine Annealing 스케줄러는 num_epochs가 필요합니다.")
        
        eta_min = kwargs.get('eta_min', 0)  # 최소 학습률
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=eta_min
        )
        print(f"Cosine Annealing 스케줄러 생성 완료")
        print(f"   총 에포크: {num_epochs}")
        print(f"   최소 학습률: {eta_min}")
    
    elif scheduler_name.lower() == 'reduce_lr_on_plateau':
        # ReduceLROnPlateau: 검증 손실이 개선되지 않을 때 학습률 감소
        mode = kwargs.get('mode', 'min')  # 'min' 또는 'max'
        factor = kwargs.get('factor', 0.5)  # 학습률 감소 비율
        patience = kwargs.get('patience', 5)  # 개선 없이 기다릴 에포크 수
        threshold = kwargs.get('threshold', 1e-4)
        min_lr = kwargs.get('min_lr', 0)
        
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            threshold=threshold,
            min_lr=min_lr
        )
        print(f"ReduceLROnPlateau 스케줄러 생성 완료")
        print(f"   모드: {mode}")
        print(f"   감소 비율: {factor}")
        print(f"   인내심: {patience} 에포크")
        print(f"   최소 학습률: {min_lr}")
    
    else:
        raise ValueError(
            f"지원하지 않는 스케줄러: {scheduler_name}\n"
            f"지원 스케줄러: ['cosine_annealing', 'reduce_lr_on_plateau']"
        )
    
    return scheduler


def get_learning_rate(optimizer):
    """
    현재 학습률 조회
    
    Args:
        optimizer: 옵티마이저 객체
        
    Returns:
        lr: 현재 학습률
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']


def update_scheduler(scheduler, scheduler_name, metric_value=None):
    """
    스케줄러 업데이트
    
    Args:
        scheduler: 스케줄러 객체
        scheduler_name: 스케줄러 이름
        metric_value: 메트릭 값 (ReduceLROnPlateau용)
    """
    if scheduler_name.lower() == 'cosine_annealing':
        scheduler.step()
    elif scheduler_name.lower() == 'reduce_lr_on_plateau':
        if metric_value is None:
            raise ValueError("ReduceLROnPlateau 스케줄러는 metric_value가 필요합니다.")
        scheduler.step(metric_value)
    else:
        raise ValueError(f"지원하지 않는 스케줄러: {scheduler_name}")

