"""
손실 함수 정의 모듈
CrossEntropyLoss를 기본으로 하며, Focal Loss, Label Smoothing 등 추가 손실 함수 제공
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_loss_function(loss_name='cross_entropy', num_classes=2, **kwargs):
    """
    손실 함수 생성 함수
    
    Args:
        loss_name: 손실 함수 이름 ('cross_entropy', 'focal', 'label_smoothing')
        num_classes: 클래스 수
        **kwargs: 손실 함수별 추가 인자
        
    Returns:
        criterion: 손실 함수 객체
        
    Example:
        >>> criterion = get_loss_function('cross_entropy')
        >>> criterion = get_loss_function('focal', alpha=[0.5, 0.5], gamma=2.0)
        >>> criterion = get_loss_function('label_smoothing', num_classes=2, smoothing=0.1)
    """
    if loss_name.lower() == 'cross_entropy':
        # 클래스 가중치 설정 (불균형 데이터셋용)
        weight = kwargs.get('weight', None)
        if weight is not None:
            weight = torch.tensor(weight, dtype=torch.float32)
        criterion = nn.CrossEntropyLoss(weight=weight)
    
    elif loss_name.lower() == 'focal':
        alpha = kwargs.get('alpha', None)
        gamma = kwargs.get('gamma', 2.0)
        criterion = FocalLoss(alpha=alpha, gamma=gamma)
    
    elif loss_name.lower() == 'label_smoothing':
        smoothing = kwargs.get('smoothing', 0.1)
        criterion = LabelSmoothingLoss(num_classes=num_classes, smoothing=smoothing)
    
    else:
        raise ValueError(
            f"지원하지 않는 손실 함수: {loss_name}\n"
            f"지원 함수: ['cross_entropy', 'focal', 'label_smoothing']"
        )
    
    print(f"✅ 손실 함수 생성 완료: {loss_name}")
    return criterion


class FocalLoss(nn.Module):
    """
    Focal Loss 구현
    
    불균형 데이터셋에서 효과적인 손실 함수
    
    Args:
        alpha: 클래스별 가중치
        gamma: 포커싱 파라미터
    """
    def __init__(self, alpha=None, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        """순전파"""
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing Loss 구현
    
    Args:
        num_classes: 클래스 수
        smoothing: 스무딩 파라미터
    """
    def __init__(self, num_classes, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
    
    def forward(self, inputs, targets):
        """순전파"""
        log_probs = F.log_softmax(inputs, dim=1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, targets.data.unsqueeze(1), 1.0 - self.smoothing)
        
        return torch.mean(torch.sum(-true_dist * log_probs, dim=1))

