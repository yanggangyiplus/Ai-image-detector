"""
랜덤 시드 고정 모듈
"""
import random
import numpy as np
import torch
import os


def set_seed(seed=42):
    """
    모든 랜덤 시드 고정
    
    Args:
        seed: 시드 값
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 재현성을 위한 추가 설정
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"시드 고정 완료: {seed}")

