"""
Streamlit 웹 데모 페이지
AI 생성 이미지와 실제 이미지를 분류하는 인터랙티브 웹 애플리케이션
"""
import streamlit as st
import torch
from PIL import Image
import sys
import os
import tempfile
from pathlib import Path
from uuid import uuid4
import plotly.graph_objects as go

# 프로젝트 루트를 경로에 추가
# app/web/web_demo.py -> app/web -> app -> 프로젝트 루트
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.inference.inference import (
    load_model_for_inference,
    predict_single_image,
)

# 페이지 설정
st.set_page_config(
    page_title="AI Image Detector",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 상수 정의
CLASS_NAMES = ['Real', 'AI']
CLASS_COLORS = {
    "Real": "#3498db",
    "AI": "#e74c3c"
}

# 체크포인트 경로 정의
CHECKPOINT_PATHS = {
    'cnn': Path('experiments/checkpoints/CNN_resnet18_best.pth'),
    'vit': Path('experiments/checkpoints/ViT_vit_base_best.pth')
}

# 모델 설정
MODEL_CONFIGS = {
    'cnn': {
        'model_type': 'cnn',
        'model_name': 'resnet18',
        'num_classes': 2
    },
    'vit': {
        'model_type': 'vit',
        'model_name': 'vit_base',
        'num_classes': 2
    }
}


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


@st.cache_resource(show_spinner=False)
def load_model(model_type='cnn', checkpoint_path=None):
    """
    모델 로드 함수 (통합)
    
    Args:
        model_type: 모델 타입 ('cnn' 또는 'vit')
        checkpoint_path: 체크포인트 파일 경로 (None이면 기본 경로 사용)
        
    Returns:
        tuple: (model, device) 또는 (None, None) (로드 실패 시)
    """
    try:
        # 체크포인트 경로 확인
        if checkpoint_path is None:
            checkpoint_path = CHECKPOINT_PATHS.get(model_type)
        
        if checkpoint_path is None or not checkpoint_path.exists():
            return None, None, f"체크포인트 파일을 찾을 수 없습니다: {checkpoint_path}"
        
        # 디바이스 선택
        device = get_device()
        
        # 모델 설정 가져오기
        config = MODEL_CONFIGS.get(model_type)
        if config is None:
            return None, None, f"지원하지 않는 모델 타입: {model_type}"
        
        # 모델 로드
        model, checkpoint = load_model_for_inference(
            checkpoint_path=checkpoint_path,
            model_type=config['model_type'],
            model_name=config['model_name'],
            num_classes=config['num_classes'],
            device=device
        )
        
        return model, device, None
        
    except FileNotFoundError as e:
        return None, None, f"체크포인트 파일을 찾을 수 없습니다: {e}"
    except Exception as e:
        return None, None, f"모델 로드 실패: {str(e)}"


def save_uploaded_image(uploaded_file):
    """
    업로드된 이미지를 임시 파일로 저장 (안전한 방식)
    
    Args:
        uploaded_file: Streamlit UploadedFile 객체
        
    Returns:
        str: 임시 파일 경로
        
    Raises:
        ValueError: 이미지를 읽을 수 없을 때
    """
    # 임시 디렉토리 생성
    temp_dir = Path('/tmp') if os.name != 'nt' else Path(tempfile.gettempdir())
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # UUID 기반 임시 파일명 생성
    file_ext = Path(uploaded_file.name).suffix or '.jpg'
    temp_path = temp_dir / f"{uuid4()}{file_ext}"
    
    # 이미지 저장 (예외 처리)
    try:
        image = Image.open(uploaded_file).convert("RGB")
        image.save(temp_path, format='JPEG' if file_ext.lower() in ['.jpg', '.jpeg'] else 'PNG')
    except Exception as e:
        raise ValueError(f"이미지를 저장할 수 없습니다: {str(e)}")
    
    return str(temp_path)


def format_prediction_result(result):
    """
    예측 결과를 명시적으로 정렬된 구조로 변환하고 타입 보장
    
    Args:
        result: predict_single_image의 반환값
        
    Returns:
        dict: 정렬된 결과 딕셔너리 (모든 숫자는 Python float 타입 보장)
    """
    # 타입 보장: Tensor나 numpy 타입을 Python float로 변환
    confidence = float(result.get("confidence", 0.0))
    probabilities = {
        k: float(v) for k, v in result.get("probabilities", {}).items()
    }
    
    return {
        "predicted_class": result["predicted_class"],
        "confidence": confidence,
        "probabilities": probabilities,
        "predicted_class_idx": int(result.get("predicted_class_idx", 0)),
        "is_ai": result.get("is_ai"),
        "image_path": result.get("image_path", "uploaded_image")
    }


def create_probability_chart(prob_data, pred_class):
    """
    확률 분포 차트 생성
    
    Args:
        prob_data: 클래스별 확률 딕셔너리
        pred_class: 예측된 클래스 이름
        
    Returns:
        plotly.graph_objects.Figure: Plotly 차트 객체
    """
    fig = go.Figure(data=[
        go.Bar(
            x=list(prob_data.keys()),
            y=list(prob_data.values()),
            marker_color=[CLASS_COLORS.get(k, "#95a5a6") for k in prob_data.keys()],
            text=[f"{v:.2%}" for v in prob_data.values()],
            textposition='auto',
        )
    ])
    fig.update_layout(
        title="예측 확률",
        xaxis_title="클래스",
        yaxis_title="확률",
        yaxis=dict(range=[0, 1]),
        height=300
    )
    return fig


def handle_prediction_error(error, error_type="일반"):
    """
    예측 중 발생한 오류 처리
    
    Args:
        error: Exception 객체
        error_type: 오류 타입 설명
    """
    error_msg = str(error)
    
    if "CUDA" in error_msg or "cuda" in error_msg:
        st.error("GPU 메모리 부족 또는 CUDA 오류가 발생했습니다. CPU 모드로 다시 시도해주세요.")
    elif "format" in error_msg.lower() or "decode" in error_msg.lower():
        st.error("이미지 포맷 오류: 지원하지 않는 이미지 형식입니다. (PNG, JPG, JPEG만 지원)")
    elif "memory" in error_msg.lower():
        st.error("메모리 부족: 이미지 크기가 너무 큽니다. 더 작은 이미지로 시도해주세요.")
    else:
        st.error(f"예측 중 오류 발생 ({error_type}): {error_msg}")


def render_model_status(model, device, error_msg=None):
    """
    모델 상태 UI 렌더링
    
    Args:
        model: 로드된 모델 (None일 수 있음)
        device: 사용 중인 디바이스
        error_msg: 오류 메시지 (있는 경우)
    """
    if model is None:
        if error_msg:
            st.error(f"⚠️ 모델 로드 실패: {error_msg}")
        else:
            st.warning("⚠️ 모델이 로드되지 않았습니다.")
            st.info("💡 체크포인트 파일이 `experiments/checkpoints/` 디렉토리에 있는지 확인해주세요.")
        
        # GPU 권장 메시지
        if device == 'cpu':
            st.info("💻 GPU를 사용하면 추론 속도가 향상됩니다.")
    else:
        device_emoji = "🚀" if device != 'cpu' else "💻"
        st.success(f"✅ 모델 로드 완료 ({device_emoji} {device.upper()})")


# 제목 및 설명
st.title("🖼️ AI Image Detector")
st.markdown("""
### 딥러닝 기반 AI 생성 이미지 탐지 시스템

이 애플리케이션은 **CNN (ResNet18)** 및 **Vision Transformer (ViT)** 모델을 사용하여 
AI 생성 이미지와 실제 이미지를 구분합니다.

**사용 방법**: 사이드바에서 이미지를 업로드하고 모델을 선택한 후 예측 버튼을 클릭하세요.
""")

# 사이드바 설정
st.sidebar.header("⚙️ 설정")

# 모델 선택
model_type_radio = st.sidebar.radio(
    "모델 선택",
    ["CNN (ResNet18)", "ViT (Vision Transformer)"],
    help="사용할 모델을 선택하세요"
)

# 모델 타입 매핑
model_type_key = 'cnn' if model_type_radio == "CNN (ResNet18)" else 'vit'

# 모델 로드
with st.sidebar:
    with st.spinner(f"{model_type_radio} 모델 로드 중..."):
        model, device, error_msg = load_model(model_type=model_type_key)
        render_model_status(model, device, error_msg)

# 이미지 업로드
st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader(
    "📤 이미지 업로드",
    type=['png', 'jpg', 'jpeg', 'bmp'],
    help="분석할 이미지를 업로드하세요"
)

# 메인 영역
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📸 입력 이미지")
    
    if uploaded_file is not None:
        try:
            # 이미지 열기 및 RGB 변환 (예외 처리 강화)
            try:
                image = Image.open(uploaded_file).convert("RGB")
            except Exception as img_error:
                st.error("이미지를 읽는 중 오류가 발생했습니다. 다른 이미지를 시도해주세요.")
                st.error(f"오류 상세: {str(img_error)}")
                st.stop()
            
            st.image(image, caption="업로드된 이미지", use_container_width=True)
            
            # 이미지 정보
            st.info(f"**이미지 크기**: {image.size[0]} × {image.size[1]} pixels")
            
            # 예측 버튼
            if model is not None:
                # 중복 클릭 방지 체크
                if st.session_state.get("predicting", False):
                    st.warning("예측 중입니다. 잠시만 기다려주세요.")
                    st.stop()
                
                if st.button("🔍 예측하기", type="primary", use_container_width=True):
                    temp_path = None
                    try:
                        # 예측 시작 플래그 설정
                        st.session_state["predicting"] = True
                        
                        # 임시 파일로 저장 (안전한 방식)
                        temp_path = save_uploaded_image(uploaded_file)
                        
                        # 예측 수행
                        result = predict_single_image(
                            model=model,
                            image_path=temp_path,
                            device=device,
                            class_names=CLASS_NAMES
                        )
                        
                        # 타입 보장된 결과로 변환
                        formatted_result = format_prediction_result(result)
                        
                        # 결과를 session state에 저장
                        st.session_state['prediction_result'] = formatted_result
                        st.session_state['image'] = image
                        
                    except ValueError as e:
                        # 이미지 포맷 오류
                        handle_prediction_error(e, "이미지 포맷 오류")
                    except RuntimeError as e:
                        # 메모리 또는 GPU 오류
                        handle_prediction_error(e, "런타임 오류")
                    except Exception as e:
                        # 기타 오류
                        handle_prediction_error(e, "예측 오류")
                    finally:
                        # 예측 완료 플래그 해제
                        st.session_state["predicting"] = False
                        
                        # 임시 파일 삭제
                        if temp_path and os.path.exists(temp_path):
                            try:
                                os.unlink(temp_path)
                            except Exception:
                                pass  # 삭제 실패해도 계속 진행
            else:
                st.warning("⚠️ 모델이 로드되지 않았습니다. 체크포인트 파일을 확인해주세요.")
                
        except Exception as e:
            st.error(f"이미지 로드 실패: {str(e)}")
    else:
        st.info("👈 사이드바에서 이미지를 업로드하세요")

with col2:
    st.header("📊 예측 결과")
    
    if 'prediction_result' in st.session_state:
        result = st.session_state['prediction_result']
        
        # 타입 보장: 이미 format_prediction_result에서 처리되었지만 안전을 위해 재확인
        pred_class = result['predicted_class']
        confidence = float(result.get('confidence', 0.0))
        prob_data = {k: float(v) for k, v in result.get('probabilities', {}).items()}
        
        # 결과 카드
        if pred_class == 'AI':
            st.error(f"🤖 **AI 생성 이미지**로 판단되었습니다.")
        else:
            st.success(f"📷 **실제 이미지**로 판단되었습니다.")
        
        # 신뢰도 표시
        st.metric("신뢰도", f"{confidence:.2%}")
        
        # 진행 바
        st.progress(confidence)
        
        # 확률 분포 시각화
        st.subheader("클래스별 확률 분포")
        
        fig = create_probability_chart(prob_data, pred_class)
        st.plotly_chart(fig, use_container_width=True)
        
        # 상세 정보
        with st.expander("📋 상세 정보"):
            st.json(result)
        
        # 통계 정보
        st.subheader("📈 통계")
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("예측 클래스", pred_class)
        with col_b:
            st.metric("클래스 인덱스", result['predicted_class_idx'])
        
    else:
        st.info("이미지를 업로드하고 예측 버튼을 클릭하세요.")

# 푸터
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>AI Image Detector | Powered by PyTorch & Streamlit</p>
    <p>CNN (ResNet18) & Vision Transformer (ViT-Base) 모델 사용</p>
</div>
""", unsafe_allow_html=True)
