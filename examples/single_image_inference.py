#!/usr/bin/env python3
"""
단일 이미지 추론 예제
사용법: python examples/single_image_inference.py <이미지_경로> [--model CNN|ViT] [--checkpoint 경로]
"""
import sys
import argparse
from pathlib import Path

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.inference.inference import (
    load_model_for_inference,
    predict_single_image,
    print_prediction_result,
    save_prediction_result
)


def main():
    parser = argparse.ArgumentParser(description='단일 이미지 추론')
    parser.add_argument('image_path', type=str, help='추론할 이미지 경로')
    parser.add_argument('--model', type=str, choices=['CNN', 'ViT'], default='CNN',
                       help='사용할 모델 (CNN 또는 ViT)')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='체크포인트 파일 경로 (기본값: 자동 탐지)')
    parser.add_argument('--model_name', type=str, default=None,
                       help='모델 이름 (예: resnet18, vit_base)')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda', 'mps'],
                       help='사용할 디바이스')
    parser.add_argument('--save', type=str, default=None,
                       help='결과를 저장할 JSON 파일 경로')
    parser.add_argument('--no-verbose', action='store_true',
                       help='상세 정보 출력 안 함')
    
    args = parser.parse_args()
    
    # 디바이스 설정
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = args.device
    
    print(f"디바이스: {device}")
    
    # 체크포인트 경로 설정
    if args.checkpoint is None:
        checkpoint_dir = Path('experiments/checkpoints')
        if args.model == 'CNN':
            checkpoint_path = checkpoint_dir / 'CNN_resnet18_best.pth'
            model_type = 'cnn'
            model_name = args.model_name or 'resnet18'
        else:  # ViT
            checkpoint_path = checkpoint_dir / 'ViT_vit_base_best.pth'
            model_type = 'vit'
            model_name = args.model_name or 'vit_base'
    else:
        checkpoint_path = Path(args.checkpoint)
        model_type = 'cnn' if args.model == 'CNN' else 'vit'
        model_name = args.model_name or ('resnet18' if args.model == 'CNN' else 'vit_base')
    
    if not checkpoint_path.exists():
        print(f"❌ 체크포인트 파일을 찾을 수 없습니다: {checkpoint_path}")
        sys.exit(1)
    
    # 모델 로드
    print(f"\n모델 로드 중...")
    model, checkpoint = load_model_for_inference(
        checkpoint_path=checkpoint_path,
        model_type=model_type,
        model_name=model_name,
        num_classes=2,
        device=device
    )
    
    # 클래스 이름
    class_names = ['Real', 'AI']
    
    # 이미지 추론
    print(f"\n이미지 추론 중...")
    result = predict_single_image(
        model=model,
        image_path=args.image_path,
        device=device,
        class_names=class_names
    )
    
    # 결과 출력
    print_prediction_result(result, verbose=not args.no_verbose)
    
    # 결과 저장
    if args.save:
        save_prediction_result(result, args.save)


if __name__ == '__main__':
    main()


