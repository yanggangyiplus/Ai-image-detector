"""
모델 유틸리티 함수 (로더, 가중치 저장 등)
"""
import torch
import os
from pathlib import Path
from .cnn import (
    ResNet18Classifier, ResNet50Classifier,
    EfficientNetB0Classifier, EfficientNetB2Classifier,
    MobileNetV3SmallClassifier
)
from .vit import ViTBaseClassifier, ViTForImageClassificationWrapper


def create_cnn_model(model_name='resnet18', num_classes=2, pretrained=True, **kwargs):
    """
    CNN 모델 생성 함수
    
    Args:
        model_name: 모델 이름 ('resnet18', 'resnet50', 'efficientnet_b0', 
                              'efficientnet_b2', 'mobilenet_v3_small')
        num_classes: 분류할 클래스 수
        pretrained: 사전 훈련된 가중치 사용 여부
        **kwargs: 모델별 추가 인자
        
    Returns:
        model: 생성된 모델 객체
        
    Example:
        >>> model = create_cnn_model('resnet18', num_classes=2, pretrained=True)
        >>> model = create_cnn_model('efficientnet_b0', num_classes=2)
    """
    model_registry = {
        'resnet18': ResNet18Classifier,
        'resnet50': ResNet50Classifier,
        'efficientnet_b0': EfficientNetB0Classifier,
        'efficientnet_b2': EfficientNetB2Classifier,
        'mobilenet_v3_small': MobileNetV3SmallClassifier,
    }
    
    if model_name.lower() not in model_registry:
        raise ValueError(
            f"지원하지 않는 모델: {model_name}\n"
            f"지원 모델: {list(model_registry.keys())}"
        )
    
    model_class = model_registry[model_name.lower()]
    model = model_class(num_classes=num_classes, pretrained=pretrained, **kwargs)
    
    print(f"{model_name} 모델 생성 완료 (클래스 수: {num_classes}, 사전훈련: {pretrained})")
    return model


def create_vit_model(model_name='vit_base', num_classes=2, pretrained=True, 
                     freeze_backbone=False, freeze_layers=None, **kwargs):
    """
    Vision Transformer 모델 생성 함수
    
    Args:
        model_name: 모델 이름 ('vit_base', 'vit_for_classification')
        num_classes: 분류할 클래스 수
        pretrained: 사전 훈련된 가중치 사용 여부
        freeze_backbone: 백본 레이어 고정 여부 (Fine-tuning)
        freeze_layers: 고정할 레이어 수
        **kwargs: 모델별 추가 인자
        
    Returns:
        model: 생성된 모델 객체
        
    Example:
        >>> model = create_vit_model('vit_base', num_classes=2, pretrained=True)
        >>> model = create_vit_model('vit_base', freeze_backbone=True, freeze_layers=6)
    """
    if model_name.lower() == 'vit_base':
        model = ViTBaseClassifier(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            freeze_layers=freeze_layers,
            **kwargs
        )
    elif model_name.lower() == 'vit_for_classification':
        model = ViTForImageClassificationWrapper(
            num_classes=num_classes,
            pretrained=pretrained,
            **kwargs
        )
    else:
        raise ValueError(
            f"지원하지 않는 모델: {model_name}\n"
            f"지원 모델: ['vit_base', 'vit_for_classification']"
        )
    
    print(f"{model_name} 모델 생성 완료 (클래스 수: {num_classes}, 사전훈련: {pretrained})")
    return model


def save_model(model, save_path, epoch=None, optimizer=None, metrics=None, 
               model_config=None):
    """
    모델 가중치 저장
    
    Args:
        model: 저장할 모델
        save_path: 저장 경로
        epoch: 에포크 번호
        optimizer: 옵티마이저 (선택)
        metrics: 메트릭 딕셔너리 (선택)
        model_config: 모델 설정 딕셔너리 (선택)
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if metrics is not None:
        checkpoint['metrics'] = metrics
    
    if model_config is not None:
        checkpoint['model_config'] = model_config
    
    # 디렉토리 생성
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(checkpoint, save_path)
    print(f"모델 저장 완료: {save_path}")


def load_model(model, load_path, device='cpu', strict=True):
    """
    모델 가중치 로드
    
    Args:
        model: 가중치를 로드할 모델
        load_path: 로드할 체크포인트 경로
        device: 디바이스 ('cpu' 또는 'cuda')
        strict: strict 모드 (모든 키가 일치해야 함)
        
    Returns:
        checkpoint: 로드된 체크포인트 딕셔너리
    """
    load_path = Path(load_path)
    if not load_path.exists():
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {load_path}")
    
    checkpoint = torch.load(load_path, map_location=device)
    
    # 모델 가중치 로드
    if strict:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    print(f"모델 로드 완료: {load_path}")
    if 'epoch' in checkpoint:
        print(f"   에포크: {checkpoint['epoch']}")
    if 'metrics' in checkpoint:
        print(f"   메트릭: {checkpoint['metrics']}")
    
    return checkpoint


def load_model_from_checkpoint(checkpoint_path, model_type='cnn', model_name='resnet18',
                               num_classes=2, device='cpu', **kwargs):
    """
    체크포인트에서 모델을 생성하고 로드하는 편의 함수
    
    Args:
        checkpoint_path: 체크포인트 파일 경로
        model_type: 모델 타입 ('cnn' 또는 'vit')
        model_name: 모델 이름
        num_classes: 클래스 수
        device: 디바이스
        **kwargs: 모델 생성에 필요한 추가 인자
        
    Returns:
        model: 로드된 모델
        checkpoint: 체크포인트 정보
    """
    # 모델 생성
    if model_type.lower() == 'cnn':
        model = create_cnn_model(model_name, num_classes=num_classes, **kwargs)
    elif model_type.lower() == 'vit':
        model = create_vit_model(model_name, num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"지원하지 않는 모델 타입: {model_type}")
    
    # 가중치 로드
    checkpoint = load_model(model, checkpoint_path, device=device)
    
    model.to(device)
    model.eval()
    
    return model, checkpoint


def get_model_summary(model, input_size=(3, 224, 224)):
    """
    모델 구조 요약 출력
    
    Args:
        model: 모델 객체
        input_size: 입력 텐서 크기
    """
    try:
        from torchsummary import summary
        summary(model, input_size)
    except ImportError:
        print("torchsummary가 설치되지 않았습니다. 'pip install torchsummary'로 설치하세요.")
        # 대안: 모델 구조 직접 출력
        print("\n모델 구조:")
        print(model)


def count_parameters(model, trainable_only=True):
    """
    모델 파라미터 수 계산
    
    Args:
        model: 모델 객체
        trainable_only: 학습 가능한 파라미터만 계산할지 여부
        
    Returns:
        total_params: 총 파라미터 수
    """
    if trainable_only:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        total_params = sum(p.numel() for p in model.parameters())
    
    return total_params


def print_model_info(model, input_size=(3, 224, 224)):
    """
    모델 정보 출력
    
    Args:
        model: 모델 객체
        input_size: 입력 텐서 크기
    """
    print("=" * 60)
    print("모델 정보")
    print("=" * 60)
    print(f"모델 타입: {type(model).__name__}")
    print(f"총 파라미터 수: {count_parameters(model, trainable_only=False):,}")
    print(f"학습 가능 파라미터 수: {count_parameters(model, trainable_only=True):,}")
    print("=" * 60)

