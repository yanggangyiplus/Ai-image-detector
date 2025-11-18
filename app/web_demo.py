"""
Streamlit 웹 데모 페이지
AI 생성 이미지와 실제 이미지를 분류하는 인터랙티브 웹 애플리케이션
"""
import streamlit as st
import torch
from PIL import Image
import sys
import tempfile
import os
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.inference import (
    load_model_for_inference,
    predict_single_image,
    print_prediction_result
)

# 페이지 설정
st.set_page_config(
    page_title="AI Image Detector",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 제목 및 설명
st.title("🖼️ AI Image Detector")
st.markdown("""
### 딥러닝 기반 AI 생성 이미지 탐지 시스템

이 애플리케이션은 **CNN (ResNet18)** 및 **Vision Transformer (ViT)** 모델을 사용하여 
AI 생성 이미지와 실제 이미지를 구분합니다.

**사용 방법**: 사이드바에서 이미지를 업로드하고 모델을 선택한 후 예측 버튼을 클릭하세요.
""")

# 모델 로드 함수 (캐싱)
@st.cache_resource
def load_cnn_model():
    """CNN 모델 로드"""
    try:
        checkpoint_path = Path('experiments/checkpoints/CNN_resnet18_best.pth')
        if not checkpoint_path.exists():
            return None, None
        
        device = 'cuda' if torch.cuda.is_available() else 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu'
        model, checkpoint = load_model_for_inference(
            checkpoint_path=checkpoint_path,
            model_type='cnn',
            model_name='resnet18',
            num_classes=2,
            device=device
        )
        return model, device
    except Exception as e:
        st.error(f"CNN 모델 로드 실패: {e}")
        return None, None

@st.cache_resource
def load_vit_model():
    """ViT 모델 로드"""
    try:
        checkpoint_path = Path('experiments/checkpoints/ViT_vit_base_best.pth')
        if not checkpoint_path.exists():
            return None, None
        
        device = 'cuda' if torch.cuda.is_available() else 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu'
        model, checkpoint = load_model_for_inference(
            checkpoint_path=checkpoint_path,
            model_type='vit',
            model_name='vit_base',
            num_classes=2,
            device=device
        )
        return model, device
    except Exception as e:
        st.error(f"ViT 모델 로드 실패: {e}")
        return None, None

# 사이드바 설정
st.sidebar.header("⚙️ 설정")

# 모델 선택
model_type = st.sidebar.radio(
    "모델 선택",
    ["CNN (ResNet18)", "ViT (Vision Transformer)"],
    help="사용할 모델을 선택하세요"
)

# 모델 로드
if model_type == "CNN (ResNet18)":
    with st.sidebar:
        with st.spinner("CNN 모델 로드 중..."):
            model, device = load_cnn_model()
            if model is not None:
                st.success("✅ CNN 모델 로드 완료")
else:
    with st.sidebar:
        with st.spinner("ViT 모델 로드 중..."):
            model, device = load_vit_model()
            if model is not None:
                st.success("✅ ViT 모델 로드 완료")

# 이미지 업로드
st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader(
    "📤 이미지 업로드",
    type=['png', 'jpg', 'jpeg', 'bmp'],
    help="분석할 이미지를 업로드하세요"
)

# 클래스 이름
class_names = ['Real', 'AI']

# 메인 영역
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📸 입력 이미지")
    
    if uploaded_file is not None:
        # 이미지 표시
        image = Image.open(uploaded_file)
        st.image(image, caption="업로드된 이미지", use_container_width=True)
        
        # 이미지 정보
        st.info(f"**이미지 크기**: {image.size[0]} × {image.size[1]} pixels")
        
        # 예측 버튼
        if model is not None:
            if st.button("🔍 예측하기", type="primary", use_container_width=True):
                with st.spinner("예측 중..."):
                    # 임시 파일로 저장
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                        image.save(tmp_file.name)
                        temp_path = tmp_file.name
                    
                    try:
                        # 예측 수행
                        result = predict_single_image(
                            model=model,
                            image_path=temp_path,
                            device=device,
                            class_names=class_names
                        )
                        
                        # 결과를 session state에 저장
                        st.session_state['prediction_result'] = result
                        st.session_state['image'] = image
                        
                    finally:
                        # 임시 파일 삭제
                        if os.path.exists(temp_path):
                            os.unlink(temp_path)
        else:
            st.warning("⚠️ 모델이 로드되지 않았습니다. 체크포인트 파일을 확인해주세요.")
    else:
        st.info("👈 사이드바에서 이미지를 업로드하세요")
        # 샘플 이미지 표시
        st.markdown("### 샘플 이미지")
        st.markdown("테스트 데이터셋에서 샘플 이미지를 확인할 수 있습니다.")

with col2:
    st.header("📊 예측 결과")
    
    if 'prediction_result' in st.session_state:
        result = st.session_state['prediction_result']
        
        # 예측 클래스 및 신뢰도
        pred_class = result['predicted_class']
        confidence = result['confidence']
        
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
        
        # Plotly를 사용한 시각화
        prob_data = result['probabilities']
        fig = go.Figure(data=[
            go.Bar(
                x=list(prob_data.keys()),
                y=list(prob_data.values()),
                marker_color=['#2ecc71' if k == pred_class else '#e74c3c' for k in prob_data.keys()],
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

