"""
단일 이미지 추론 코드 
"""
import torch
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
import json
import sys
import os

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

try:
    from src.data.preprocess import get_test_transforms
except ImportError:
    from data.preprocess import get_test_transforms


def load_image(image_path, transform=None, image_size=224):
    """
    이미지 로드 및 전처리
    
    Args:
        image_path: 이미지 파일 경로 (str 또는 Path) 또는 PIL Image 객체
        transform: 전처리 변환 함수 (None이면 기본 변환 사용)
        image_size: 이미지 크기
        
    Returns:
        image_tensor: 전처리된 이미지 텐서 (1, C, H, W)
        
    Raises:
        FileNotFoundError: 이미지 파일이 존재하지 않을 때
        ValueError: 이미지를 로드할 수 없을 때
    """
    # PIL Image 객체인 경우 직접 사용
    if isinstance(image_path, Image.Image):
        image = image_path.convert('RGB')
    else:
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")
        
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}, 오류: {e}")
    
    # 기본 transform 사용
    if transform is None:
        transform = get_test_transforms(image_size)
    
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor


def predict_single_image(model, image_path, device='cpu', class_names=None, return_tensor=False):
    """
    단일 이미지에 대한 예측 수행
    
    Args:
        model: 학습된 모델 (torch.nn.Module)
        image_path: 이미지 파일 경로 (str 또는 Path) 또는 PIL Image 객체
        device: 디바이스 ('cpu', 'cuda', 'mps')
        class_names: 클래스 이름 리스트 (예: ['Real', 'AI'])
        return_tensor: 원본 텐서도 반환할지 여부
        
    Returns:
        result: 예측 결과 딕셔너리
            - image_path: 이미지 경로 또는 'uploaded_image'
            - predicted_class: 예측된 클래스 이름 또는 인덱스
            - predicted_class_idx: 예측된 클래스 인덱스
            - confidence: 예측 신뢰도 (확률)
            - probabilities: 모든 클래스에 대한 확률 딕셔너리
            - is_ai: AI 이미지 여부 (True/False)
            - image_tensor: 원본 텐서 (return_tensor=True일 때만)
    """
    model.eval()
    
    # 이미지 로드
    image_tensor = load_image(image_path)
    image_tensor = image_tensor.to(device)
    
    # 예측 수행
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs.data, 1)
    
    # 결과 구성
    pred_class_idx = predicted.item()
    pred_prob = probabilities[0][pred_class_idx].item()
    
    # 클래스 이름 처리
    if class_names is None:
        class_names = [f'Class_{i}' for i in range(len(probabilities[0]))]
    
    # 이미지 경로 처리 (PIL Image인 경우)
    image_path_str = 'uploaded_image' if isinstance(image_path, Image.Image) else str(image_path)
    
    result = {
        'image_path': image_path_str,
        'predicted_class': class_names[pred_class_idx],
        'predicted_class_idx': pred_class_idx,
        'confidence': float(pred_prob),
        'probabilities': {
            class_names[i]: float(probabilities[0][i].item())
            for i in range(len(probabilities[0]))
        },
        'is_ai': pred_class_idx == 1 if len(class_names) == 2 else None
    }
    
    if return_tensor:
        result['image_tensor'] = image_tensor.cpu()
    
    return result


def load_model_for_inference(checkpoint_path, model_type='cnn', model_name='resnet18', 
                             num_classes=2, device='cpu'):
    """
    추론을 위한 모델 로드
    
    Args:
        checkpoint_path: 체크포인트 파일 경로
        model_type: 모델 타입 ('cnn' 또는 'vit')
        model_name: 모델 이름 ('resnet18', 'vit_base' 등)
        num_classes: 클래스 수
        device: 디바이스
        
    Returns:
        model: 로드된 모델
        checkpoint: 체크포인트 정보
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"체크포인트 파일을 찾을 수 없습니다: {checkpoint_path}")
    
    # 모델 생성
    try:
        from src.models.model_utils import create_cnn_model, create_vit_model
    except ImportError:
        from models.model_utils import create_cnn_model, create_vit_model
    
    if model_type.lower() == 'cnn':
        model = create_cnn_model(model_name, num_classes=num_classes, pretrained=False)
    elif model_type.lower() == 'vit':
        model = create_vit_model(model_name, num_classes=num_classes, pretrained=False)
    else:
        raise ValueError(f"지원하지 않는 모델 타입: {model_type}")
    
    # 체크포인트 로드
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"모델 로드 완료: {checkpoint_path}")
    print(f"   모델 타입: {model_type.upper()}")
    print(f"   모델 이름: {model_name}")
    print(f"   Best Epoch: {checkpoint.get('epoch', 'N/A')}")
    
    return model, checkpoint


def print_prediction_result(result, verbose=True):
    """
    예측 결과를 보기 좋게 출력
    
    Args:
        result: predict_single_image의 반환값
        verbose: 상세 정보 출력 여부
    """
    print("=" * 60)
    print("📸 이미지 추론 결과")
    print("=" * 60)
    print(f"이미지 경로: {result['image_path']}")
    print(f"\n예측 결과:")
    print(f"  클래스: {result['predicted_class']}")
    print(f"  신뢰도: {result['confidence']:.4f} ({result['confidence']*100:.2f}%)")
    
    if result['is_ai'] is not None:
        status = "🤖 AI 생성 이미지" if result['is_ai'] else "📷 실제 이미지"
        print(f"  판단: {status}")
    
    if verbose:
        print(f"\n모든 클래스 확률:")
        for class_name, prob in result['probabilities'].items():
            bar_length = int(prob * 30)
            bar = "█" * bar_length + "░" * (30 - bar_length)
            print(f"  {class_name:15s}: {prob:.4f} ({prob*100:6.2f}%) {bar}")
    
    print("=" * 60)


def save_prediction_result(result, save_path):
    """
    예측 결과를 JSON 파일로 저장
    
    Args:
        result: predict_single_image의 반환값
        save_path: 저장할 파일 경로
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 텐서 제거 (JSON 직렬화 불가)
    result_to_save = {k: v for k, v in result.items() if k != 'image_tensor'}
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(result_to_save, f, indent=2, ensure_ascii=False)
    
    print(f"예측 결과 저장: {save_path}")

