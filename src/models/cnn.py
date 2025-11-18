"""
CNN 모델 정의
EfficientNet, ResNet, MobileNet 등 다양한 CNN 아키텍처 제공
"""
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import efficientnet_b0, efficientnet_b2
from torchvision.models import mobilenet_v3_small


class SimpleCNN(nn.Module):
    """
    간단한 CNN 모델
    
    Args:
        num_classes: 분류할 클래스 수
    """
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """순전파"""
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        
        x = x.view(-1, 128 * 28 * 28)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x


class ResNet18Classifier(nn.Module):
    """
    ResNet-18 기반 분류 모델
    
    Args:
        num_classes: 분류할 클래스 수
        pretrained: 사전 훈련된 가중치 사용 여부
    """
    def __init__(self, num_classes=2, pretrained=True):
        super(ResNet18Classifier, self).__init__()
        # 최신 torchvision은 weights 파라미터 사용
        if pretrained:
            self.model = models.resnet18(weights='DEFAULT')
        else:
            self.model = models.resnet18(weights=None)
        # 마지막 FC 레이어를 num_classes에 맞게 변경
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    
    def forward(self, x):
        """순전파"""
        return self.model(x)


class ResNet50Classifier(nn.Module):
    """
    ResNet-50 기반 분류 모델
    
    Args:
        num_classes: 분류할 클래스 수
        pretrained: 사전 훈련된 가중치 사용 여부
    """
    def __init__(self, num_classes=2, pretrained=True):
        super(ResNet50Classifier, self).__init__()
        # 최신 torchvision은 weights 파라미터 사용
        if pretrained:
            self.model = models.resnet50(weights='DEFAULT')
        else:
            self.model = models.resnet50(weights=None)
        # 마지막 FC 레이어를 num_classes에 맞게 변경
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    
    def forward(self, x):
        """순전파"""
        return self.model(x)


class EfficientNetB0Classifier(nn.Module):
    """
    EfficientNet-B0 기반 분류 모델
    
    Args:
        num_classes: 분류할 클래스 수
        pretrained: 사전 훈련된 가중치 사용 여부
    """
    def __init__(self, num_classes=2, pretrained=True):
        super(EfficientNetB0Classifier, self).__init__()
        if pretrained:
            self.model = efficientnet_b0(weights='DEFAULT')
        else:
            self.model = efficientnet_b0(weights=None)
        
        # 마지막 분류 레이어 변경
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(self.model.classifier[1].in_features, num_classes)
        )
    
    def forward(self, x):
        """순전파"""
        return self.model(x)


class EfficientNetB2Classifier(nn.Module):
    """
    EfficientNet-B2 기반 분류 모델
    
    Args:
        num_classes: 분류할 클래스 수
        pretrained: 사전 훈련된 가중치 사용 여부
    """
    def __init__(self, num_classes=2, pretrained=True):
        super(EfficientNetB2Classifier, self).__init__()
        if pretrained:
            self.model = efficientnet_b2(weights='DEFAULT')
        else:
            self.model = efficientnet_b2(weights=None)
        
        # 마지막 분류 레이어 변경
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(self.model.classifier[1].in_features, num_classes)
        )
    
    def forward(self, x):
        """순전파"""
        return self.model(x)


class MobileNetV3SmallClassifier(nn.Module):
    """
    MobileNet V3 Small 기반 분류 모델
    
    Args:
        num_classes: 분류할 클래스 수
        pretrained: 사전 훈련된 가중치 사용 여부
    """
    def __init__(self, num_classes=2, pretrained=True):
        super(MobileNetV3SmallClassifier, self).__init__()
        if pretrained:
            self.model = mobilenet_v3_small(weights='DEFAULT')
        else:
            self.model = mobilenet_v3_small(weights=None)
        
        # 마지막 분류 레이어 변경
        self.model.classifier = nn.Sequential(
            nn.Linear(self.model.classifier[0].in_features, 512),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        """순전파"""
        return self.model(x)


# 하위 호환성을 위한 별칭
ResNetClassifier = ResNet50Classifier

