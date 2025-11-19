"""
Vision Transformer 모델 정의
HuggingFace Transformers를 사용한 ViT 모델 및 Fine-tuning 구조
"""
import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig, ViTImageProcessor
from transformers import ViTForImageClassification


class ViTBaseClassifier(nn.Module):
    """
    HuggingFace ViT-Base 기반 분류 모델 (Fine-tuning 구조)
    
    Args:
        num_classes: 분류할 클래스 수
        model_name: 사용할 ViT 모델 이름 (기본값: 'google/vit-base-patch16-224')
        pretrained: 사전 훈련된 가중치 사용 여부
        freeze_backbone: 백본(인코더) 레이어 고정 여부 (Fine-tuning 시 유용)
        freeze_layers: 고정할 레이어 수 (None이면 freeze_backbone에 따라 결정)
    """
    def __init__(self, num_classes=2, model_name='google/vit-base-patch16-224', 
                 pretrained=True, freeze_backbone=False, freeze_layers=None):
        super(ViTBaseClassifier, self).__init__()
        
        if pretrained:
            # 사전 훈련된 모델 로드
            self.vit = ViTModel.from_pretrained(model_name)
        else:
            # 랜덤 초기화
            config = ViTConfig.from_pretrained(model_name)
            self.vit = ViTModel(config)
        
        # 분류 헤드 구성
        hidden_size = self.vit.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        # Fine-tuning 설정: 백본 일부 레이어 고정
        if freeze_backbone:
            if freeze_layers is None:
                # 기본적으로 절반의 레이어만 고정
                freeze_layers = len(self.vit.encoder.layer) // 2
            
            # 지정된 레이어 수만큼 고정
            for i in range(freeze_layers):
                for param in self.vit.encoder.layer[i].parameters():
                    param.requires_grad = False
            
            print(f"{freeze_layers}개 레이어를 고정했습니다 (Fine-tuning 모드)")
    
    def forward(self, x):
        """
        순전파
        
        Args:
            x: 입력 이미지 텐서 [batch_size, channels, height, width]
            
        Returns:
            logits: 분류 로짓 [batch_size, num_classes]
        """
        # ViT는 [batch_size, num_patches, hidden_size] 형태의 출력을 반환
        outputs = self.vit(pixel_values=x)
        
        # [CLS] 토큰의 출력 사용 (첫 번째 토큰)
        pooled_output = outputs.last_hidden_state[:, 0]
        
        # 분류 헤드를 통과
        logits = self.classifier(pooled_output)
        
        return logits


class ViTForImageClassificationWrapper(nn.Module):
    """
    HuggingFace의 ViTForImageClassification을 래핑한 모델
    더 간단한 사용을 위한 래퍼 클래스
    
    Args:
        num_classes: 분류할 클래스 수
        model_name: 사용할 ViT 모델 이름
        pretrained: 사전 훈련된 가중치 사용 여부
    """
    def __init__(self, num_classes=2, model_name='google/vit-base-patch16-224', pretrained=True):
        super(ViTForImageClassificationWrapper, self).__init__()
        
        if pretrained:
            self.model = ViTForImageClassification.from_pretrained(
                model_name,
                num_labels=num_classes,
                ignore_mismatched_sizes=True  # 클래스 수가 다를 경우 허용
            )
        else:
            config = ViTConfig.from_pretrained(model_name)
            config.num_labels = num_classes
            self.model = ViTForImageClassification(config)
    
    def forward(self, x):
        """
        순전파
        
        Args:
            x: 입력 이미지 텐서
            
        Returns:
            logits: 분류 로짓
        """
        outputs = self.model(pixel_values=x)
        return outputs.logits


# 하위 호환성을 위한 별칭
ViTClassifier = ViTBaseClassifier

