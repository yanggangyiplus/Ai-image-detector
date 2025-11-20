"""
FastAPI 백엔드 API 서버
AI 이미지 탐지를 위한 RESTful API 제공
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
import torch
from PIL import Image
import io
import tempfile
import os
from pathlib import Path
import sys
import logging
from typing import Optional, List
from pydantic import BaseModel

# 로거 설정
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 프로젝트 루트를 경로에 추가
# app/api/api.py -> app/api -> app -> 프로젝트 루트
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.inference.inference import (
    load_model_for_inference,
    predict_single_image
)

# 전역 변수
cnn_model = None
vit_model = None
cnn_device = None
vit_device = None
class_names = ["Real", "AI"]


def get_device():
    """
    사용 가능한 디바이스를 자동으로 선택
    
    Returns:
        str: 'cuda', 'mps', 또는 'cpu'
    """
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    애플리케이션 생명주기 관리 (FastAPI 0.110+ 권장 방식)
    startup과 shutdown 이벤트를 관리합니다.
    """
    # Startup: 모델 로드
    global cnn_model, vit_model, cnn_device, vit_device
    
    logger.info("모델 로드 시작...")
    
    # CNN 모델 로드
    cnn_checkpoint = Path('experiments/checkpoints/CNN_resnet18_best.pth')
    if cnn_checkpoint.exists():
        try:
            cnn_device = get_device()
            cnn_model, _ = load_model_for_inference(
                checkpoint_path=cnn_checkpoint,
                model_type='cnn',
                model_name='resnet18',
                num_classes=2,
                device=cnn_device
            )
            logger.info(f"CNN 모델 로드 완료 (device: {cnn_device})")
        except Exception as e:
            logger.error(f"CNN 모델 로드 실패: {e}")
    else:
        logger.warning(f"CNN 체크포인트 파일 없음: {cnn_checkpoint}")
    
    # ViT 모델 로드
    vit_checkpoint = Path('experiments/checkpoints/ViT_vit_base_best.pth')
    if vit_checkpoint.exists():
        try:
            vit_device = get_device()
            vit_model, _ = load_model_for_inference(
                checkpoint_path=vit_checkpoint,
                model_type='vit',
                model_name='vit_base',
                num_classes=2,
                device=vit_device
            )
            logger.info(f"ViT 모델 로드 완료 (device: {vit_device})")
        except Exception as e:
            logger.error(f"ViT 모델 로드 실패: {e}")
    else:
        logger.warning(f"ViT 체크포인트 파일 없음: {vit_checkpoint}")
    
    logger.info("모델 로드 완료!")
    
    yield
    
    # Shutdown: 리소스 정리
    logger.info("애플리케이션 종료 중...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("리소스 정리 완료")


class PredictionResponse(BaseModel):
    """예측 응답 모델"""
    image_path: str
    filename: Optional[str] = None  # 업로드된 원본 파일명
    predicted_class: str
    predicted_class_idx: int
    confidence: float
    probabilities: dict
    is_ai: bool
    model_type: str


class SkippedImage(BaseModel):
    """건너뛴 이미지 정보"""
    index: int
    filename: Optional[str] = None
    reason: str


class BatchPredictionResponse(BaseModel):
    """배치 예측 응답 모델"""
    results: List[PredictionResponse]
    skipped: List[SkippedImage] = []  # 건너뛴 이미지 목록


# FastAPI 앱 생성 (Lifespan 방식 사용)
app = FastAPI(
    title="AI Image Detector API",
    description="AI 생성 이미지와 실제 이미지를 구분하는 딥러닝 기반 API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS 설정 (웹에서 접근 가능하도록)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인만 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "message": "AI Image Detector API",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    models_status = {
        "CNN": cnn_model is not None,
        "ViT": vit_model is not None
    }
    
    return {
        "status": "healthy",
        "models": models_status,
        "devices": {
            "CNN": str(cnn_device) if cnn_device else None,
            "ViT": str(vit_device) if vit_device else None
        }
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_image(
    file: UploadFile = File(..., description="분석할 이미지 파일"),
    model_type: str = Query("cnn", description="사용할 모델 타입 (cnn 또는 vit)")
):
    """
    이미지 업로드 및 예측
    
    Args:
        file: 업로드된 이미지 파일 (jpg, png, jpeg 등)
        model_type: 사용할 모델 타입 ('cnn' 또는 'vit')
        
    Returns:
        PredictionResponse: 예측 결과 (클래스, 신뢰도, 확률 등)
    """
    # 모델 선택
    if model_type.lower() == "cnn":
        model = cnn_model
        device = cnn_device
        model_name = "CNN"
    elif model_type.lower() == "vit":
        model = vit_model
        device = vit_device
        model_name = "ViT"
    else:
        raise HTTPException(
            status_code=400,
            detail=f"지원하지 않는 모델 타입: {model_type}. 'cnn' 또는 'vit'를 사용하세요."
        )
    
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"{model_name} 모델이 로드되지 않았습니다. 체크포인트 파일을 확인해주세요."
        )
    
    # 이미지 파일 검증
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="이미지 파일만 업로드 가능합니다. 지원 형식: jpg, png, jpeg, bmp"
        )
    
    # 임시 파일 생성
    temp_file = None
    try:
        # 이미지 읽기 및 검증
        image_bytes = await file.read()
        
        # 이미지 열기 (명확한 오류 처리)
        try:
            image = Image.open(io.BytesIO(image_bytes))
        except Exception as img_error:
            raise HTTPException(
                status_code=400,
                detail=f"이미지 파일을 읽을 수 없습니다: {str(img_error)}"
            )
        
        # RGB 변환
        image = image.convert('RGB')
        
        # 임시 파일로 저장
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        image.save(temp_file.name)
        temp_path = temp_file.name
        
        # 예측 수행
        result = predict_single_image(
            model=model,
            image_path=temp_path,
            device=device,
            class_names=class_names
        )
        
        # GPU 메모리 정리 (CUDA 사용 시)
        if device == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 모델 타입 및 파일명 추가
        result['model_type'] = model_name
        result['filename'] = file.filename  # 업로드된 원본 파일명
        
        return PredictionResponse(**result)
    
    except HTTPException:
        # HTTPException은 그대로 전달
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"예측 중 오류 발생: {str(e)}"
        )
    
    finally:
        # 임시 파일 삭제
        if temp_file and os.path.exists(temp_file.name):
            try:
                os.unlink(temp_file.name)
            except Exception:
                pass  # 삭제 실패해도 계속 진행


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch_images(
    files: list[UploadFile] = File(..., description="분석할 이미지 파일 리스트"),
    model_type: str = Query("cnn", description="사용할 모델 타입 (cnn 또는 vit)")
):
    """
    여러 이미지에 대한 배치 예측
    
    Args:
        files: 업로드된 이미지 파일 리스트
        model_type: 사용할 모델 타입 ('cnn' 또는 'vit')
        
    Returns:
        BatchPredictionResponse: 예측 결과 리스트 및 건너뛴 이미지 정보
    """
    from src.inference.batch_inference import batch_predict
    
    # 모델 선택
    if model_type.lower() == "cnn":
        model = cnn_model
        device = cnn_device
        model_name = "CNN"
    elif model_type.lower() == "vit":
        model = vit_model
        device = vit_device
        model_name = "ViT"
    else:
        raise HTTPException(
            status_code=400,
            detail=f"지원하지 않는 모델 타입: {model_type}"
        )
    
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"{model_name} 모델이 로드되지 않았습니다."
        )
    
    # 임시 디렉토리 생성
    temp_dir = tempfile.mkdtemp()
    temp_paths = []
    file_mapping = {}  # temp_path -> (index, filename) 매핑
    skipped_images = []  # 건너뛴 이미지 정보
    
    try:
        # 이미지 파일 저장
        for idx, file in enumerate(files):
            # 이미지 타입 검증
            if not file.content_type or not file.content_type.startswith("image/"):
                skipped_images.append(SkippedImage(
                    index=idx,
                    filename=file.filename,
                    reason="Invalid content type (not an image)"
                ))
                logger.warning(f"이미지 {idx} ({file.filename}) 건너뜀: 잘못된 컨텐츠 타입")
                continue
            
            image_bytes = await file.read()
            
            # 이미지 열기 (명확한 오류 처리)
            try:
                image = Image.open(io.BytesIO(image_bytes))
            except Exception as img_error:
                skipped_images.append(SkippedImage(
                    index=idx,
                    filename=file.filename,
                    reason=f"Image decode failed: {str(img_error)}"
                ))
                logger.warning(f"이미지 {idx} ({file.filename}) 로드 실패: {str(img_error)}")
                continue
            
            # RGB 변환
            try:
                image = image.convert('RGB')
            except Exception as convert_error:
                skipped_images.append(SkippedImage(
                    index=idx,
                    filename=file.filename,
                    reason=f"RGB conversion failed: {str(convert_error)}"
                ))
                logger.warning(f"이미지 {idx} ({file.filename}) RGB 변환 실패: {str(convert_error)}")
                continue
            
            temp_path = os.path.join(temp_dir, f"image_{idx}.jpg")
            image.save(temp_path)
            temp_paths.append(temp_path)
            file_mapping[temp_path] = (idx, file.filename)
        
        if not temp_paths:
            raise HTTPException(status_code=400, detail="유효한 이미지 파일이 없습니다.")
        
        # 배치 예측
        results = batch_predict(
            model=model,
            image_paths=temp_paths,
            device=device,
            batch_size=min(32, len(temp_paths)),
            class_names=class_names,
            num_workers=0,
            show_progress=False
        )
        
        # GPU 메모리 정리 (CUDA 사용 시)
        if device == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 모델 타입 및 파일명 추가
        for result in results:
            result['model_type'] = model_name
            # temp_path에서 원본 파일명 찾기
            image_path = result.get('image_path', '')
            if image_path in file_mapping:
                _, filename = file_mapping[image_path]
                result['filename'] = filename
        
        return BatchPredictionResponse(
            results=[PredictionResponse(**r) for r in results],
            skipped=skipped_images
        )
    
    except HTTPException:
        # HTTPException은 그대로 전달
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"배치 예측 중 오류 발생: {str(e)}"
        )
    
    finally:
        # 임시 파일 및 디렉토리 삭제
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


@app.get("/models")
async def get_available_models():
    """사용 가능한 모델 목록 조회"""
    return {
        "available_models": {
            "CNN": cnn_model is not None,
            "ViT": vit_model is not None
        },
        "default": "cnn"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

