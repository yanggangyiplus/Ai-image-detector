"""
이미지 전처리 함수 모듈
"""
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


def get_train_transforms(image_size=224):
    """
    훈련용 이미지 전처리 변환 함수
    
    Args:
        image_size: 리사이즈할 이미지 크기
        
    Returns:
        transform: torchvision transform 객체
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])


def get_val_transforms(image_size=224):
    """
    검증용 이미지 전처리 변환 함수
    
    Args:
        image_size: 리사이즈할 이미지 크기
        
    Returns:
        transform: torchvision transform 객체
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])


def get_test_transforms(image_size=224):
    """
    테스트용 이미지 전처리 변환 함수
    
    Args:
        image_size: 리사이즈할 이미지 크기
        
    Returns:
        transform: torchvision transform 객체
    """
    return get_val_transforms(image_size)

