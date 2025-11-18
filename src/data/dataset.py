"""
커스텀 Dataset 클래스 정의
"""
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import json
from pathlib import Path


class ImageDataset(Dataset):
    """
    이미지 분류를 위한 커스텀 Dataset 클래스
    
    Args:
        data_dir: 이미지 데이터가 저장된 디렉토리 경로
        class_map: 클래스 이름과 인덱스를 매핑하는 딕셔너리
        transform: 이미지 전처리 변환 함수
    """
    def __init__(self, data_dir, class_map, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        # 클래스 매핑 로드
        if isinstance(class_map, str):
            with open(class_map, 'r') as f:
                self.class_map = json.load(f)
        else:
            self.class_map = class_map
        
        # 데이터 경로 및 레이블 수집
        self.samples = []
        self._load_samples()
    
    def _load_samples(self):
        """데이터셋에서 이미지 경로와 레이블을 수집"""
        data_path = Path(self.data_dir)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        # 클래스별 디렉토리에서 이미지 파일 찾기
        for class_name, class_idx in self.class_map.items():
            class_dir = data_path / class_name
            if not class_dir.exists():
                continue
            
            # 이미지 파일 찾기
            for img_file in class_dir.iterdir():
                if img_file.suffix.lower() in image_extensions:
                    self.samples.append({
                        'image_path': str(img_file),
                        'label': int(class_idx)
                    })
    
    def __len__(self):
        """데이터셋 크기 반환"""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        인덱스에 해당하는 이미지와 레이블 반환
        
        Args:
            idx: 데이터 인덱스
            
        Returns:
            image: 전처리된 이미지 텐서
            label: 레이블 텐서
        """
        sample = self.samples[idx]
        image_path = sample['image_path']
        label = sample['label']
        
        # 이미지 로드
        image = Image.open(image_path).convert('RGB')
        
        # 전처리 적용
        if self.transform:
            image = self.transform(image)
        
        return image, label

