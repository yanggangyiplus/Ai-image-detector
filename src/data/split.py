"""
데이터셋을 train/val/test로 분리하는 모듈
"""
import os
import shutil
from sklearn.model_selection import train_test_split
from pathlib import Path


def split_dataset(data_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
    """
    데이터셋을 train/val/test로 분리
    
    Args:
        data_dir: 원본 데이터가 있는 디렉토리
        train_ratio: 훈련 세트 비율
        val_ratio: 검증 세트 비율
        test_ratio: 테스트 세트 비율
        random_state: 랜덤 시드
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "비율의 합은 1.0이어야 합니다"
    
    data_path = Path(data_dir)
    output_dir = data_path.parent
    
    # 클래스별로 데이터 분리
    for class_name in os.listdir(data_dir):
        class_dir = data_path / class_name
        if not class_dir.is_dir():
            continue
        
        # 파일 목록 가져오기
        files = list(class_dir.glob("*"))
        files = [f for f in files if f.is_file()]
        
        # train/val/test로 분리
        train_files, temp_files = train_test_split(
            files, test_size=(val_ratio + test_ratio), random_state=random_state
        )
        
        val_files, test_files = train_test_split(
            temp_files, 
            test_size=test_ratio / (val_ratio + test_ratio),
            random_state=random_state
        )
        
        # 디렉토리 생성 및 파일 복사
        for split_name, file_list in [("train", train_files), 
                                      ("val", val_files), 
                                      ("test", test_files)]:
            split_dir = output_dir / split_name / class_name
            split_dir.mkdir(parents=True, exist_ok=True)
            
            for file_path in file_list:
                shutil.copy2(file_path, split_dir / file_path.name)
    
    print(f"데이터셋 분리 완료: train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")

