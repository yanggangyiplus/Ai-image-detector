"""
여러 이미지 한 번에 추론하는 배치 추론 코드 
"""
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import json
import sys
import os

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

try:
    from src.inference.inference import load_image
    from src.data.preprocess import get_test_transforms
except ImportError:
    from inference.inference import load_image
    from data.preprocess import get_test_transforms


class ImageInferenceDataset(Dataset):
    """
    추론용 이미지 데이터셋
    
    Args:
        image_paths: 이미지 파일 경로 리스트
        transform: 전처리 변환 함수
        image_size: 이미지 크기
    """
    def __init__(self, image_paths, transform=None, image_size=224):
        self.image_paths = [Path(p) for p in image_paths]
        self.transform = transform
        self.image_size = image_size
        
        # 존재하지 않는 이미지 필터링
        self.valid_paths = []
        self.invalid_paths = []
        for path in self.image_paths:
            if path.exists():
                self.valid_paths.append(path)
            else:
                self.invalid_paths.append(path)
        
        if self.invalid_paths:
            print(f"⚠️  {len(self.invalid_paths)}개의 이미지 파일을 찾을 수 없습니다.")
    
    def __len__(self):
        return len(self.valid_paths)
    
    def __getitem__(self, idx):
        image_path = self.valid_paths[idx]
        try:
            image_tensor = load_image(image_path, self.transform, self.image_size)
            return image_tensor.squeeze(0), str(image_path)
        except Exception as e:
            raise RuntimeError(f"이미지 로드 실패: {image_path}, 오류: {e}")


def batch_predict(model, image_paths, device='cpu', batch_size=32, class_names=None, 
                  num_workers=0, show_progress=True):
    """
    여러 이미지에 대한 배치 예측 수행
    
    Args:
        model: 학습된 모델 (torch.nn.Module)
        image_paths: 이미지 파일 경로 리스트 또는 디렉토리 경로
        device: 디바이스 ('cpu', 'cuda', 'mps')
        batch_size: 배치 크기
        class_names: 클래스 이름 리스트 (예: ['Real', 'AI'])
        num_workers: DataLoader의 worker 수 (macOS에서는 0 권장)
        show_progress: 진행 상황 표시 여부
        
    Returns:
        results: 예측 결과 리스트 (각 결과는 딕셔너리)
            - image_path: 이미지 경로
            - predicted_class: 예측된 클래스 이름
            - predicted_class_idx: 예측된 클래스 인덱스
            - confidence: 예측 신뢰도
            - probabilities: 모든 클래스에 대한 확률 딕셔너리
            - is_ai: AI 이미지 여부 (True/False)
    """
    model.eval()
    
    # 이미지 경로 처리
    if isinstance(image_paths, (str, Path)):
        image_paths = Path(image_paths)
        if image_paths.is_dir():
            # 디렉토리인 경우 모든 이미지 파일 찾기
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
            image_paths = [
                str(p) for p in image_paths.iterdir()
                if p.suffix.lower() in image_extensions
            ]
        else:
            image_paths = [str(image_paths)]
    
    if len(image_paths) == 0:
        raise ValueError("처리할 이미지가 없습니다.")
    
    print(f"📊 총 {len(image_paths)}개의 이미지 처리 시작...")
    
    # 클래스 이름 처리
    if class_names is None:
        class_names = [f'Class_{i}' for i in range(2)]  # 기본값: 2개 클래스
    
    # 데이터셋 및 데이터로더 생성
    transform = get_test_transforms()
    dataset = ImageInferenceDataset(image_paths, transform=transform)
    
    if len(dataset) == 0:
        raise ValueError("유효한 이미지 파일이 없습니다.")
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device != 'cpu' else False
    )
    
    all_results = []
    
    # 배치 처리
    iterator = tqdm(dataloader, desc="배치 추론 중") if show_progress else dataloader
    
    with torch.no_grad():
        for images, paths in iterator:
            images = images.to(device)
            
            outputs = model(images)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            # 결과 저장
            for i in range(len(paths)):
                pred_class_idx = predicted[i].item()
                pred_prob = probabilities[i][pred_class_idx].item()
                
                result = {
                    'image_path': paths[i],
                    'predicted_class': class_names[pred_class_idx],
                    'predicted_class_idx': pred_class_idx,
                    'confidence': float(pred_prob),
                    'probabilities': {
                        class_names[j]: float(probabilities[i][j].item())
                        for j in range(len(probabilities[i]))
                    },
                    'is_ai': pred_class_idx == 1 if len(class_names) == 2 else None
                }
                all_results.append(result)
    
    print(f"✅ 총 {len(all_results)}개의 이미지 처리 완료")
    
    return all_results


def save_batch_results(results, save_path, format='json'):
    """
    배치 추론 결과를 파일로 저장
    
    Args:
        results: batch_predict의 반환값
        save_path: 저장할 파일 경로
        format: 저장 형식 ('json' 또는 'csv')
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format.lower() == 'json':
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"✅ JSON 형식으로 저장: {save_path}")
    
    elif format.lower() == 'csv':
        import pandas as pd
        
        # 결과를 DataFrame으로 변환
        data = []
        for result in results:
            row = {
                'image_path': result['image_path'],
                'predicted_class': result['predicted_class'],
                'predicted_class_idx': result['predicted_class_idx'],
                'confidence': result['confidence'],
                'is_ai': result['is_ai']
            }
            # 각 클래스 확률 추가
            for class_name, prob in result['probabilities'].items():
                row[f'prob_{class_name}'] = prob
            data.append(row)
        
        df = pd.DataFrame(data)
        df.to_csv(save_path, index=False, encoding='utf-8-sig')
        print(f"✅ CSV 형식으로 저장: {save_path}")
    
    else:
        raise ValueError(f"지원하지 않는 형식: {format}")


def print_batch_summary(results, class_names=None):
    """
    배치 추론 결과 요약 출력
    
    Args:
        results: batch_predict의 반환값
        class_names: 클래스 이름 리스트
    """
    if class_names is None:
        class_names = ['Real', 'AI']
    
    total = len(results)
    ai_count = sum(1 for r in results if r.get('is_ai', False))
    real_count = total - ai_count
    
    print("\n" + "=" * 60)
    print("📊 배치 추론 결과 요약")
    print("=" * 60)
    print(f"총 처리된 이미지: {total}개")
    print(f"\n클래스별 분포:")
    print(f"  {class_names[0]}: {real_count}개 ({real_count/total*100:.2f}%)")
    print(f"  {class_names[1]}: {ai_count}개 ({ai_count/total*100:.2f}%)")
    
    if total > 0:
        avg_confidence = sum(r['confidence'] for r in results) / total
        print(f"\n평균 신뢰도: {avg_confidence:.4f} ({avg_confidence*100:.2f}%)")
        
        # 신뢰도 분포
        high_conf = sum(1 for r in results if r['confidence'] >= 0.9)
        medium_conf = sum(1 for r in results if 0.7 <= r['confidence'] < 0.9)
        low_conf = sum(1 for r in results if r['confidence'] < 0.7)
        
        print(f"\n신뢰도 분포:")
        print(f"  높음 (≥90%): {high_conf}개 ({high_conf/total*100:.2f}%)")
        print(f"  중간 (70-90%): {medium_conf}개 ({medium_conf/total*100:.2f}%)")
        print(f"  낮음 (<70%): {low_conf}개 ({low_conf/total*100:.2f}%)")
    
    print("=" * 60)

