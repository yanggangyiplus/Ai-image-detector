# 모델 배포 가이드

이 프로젝트를 다양한 플랫폼에 배포하는 방법을 안내합니다.

## 🎯 추천 플랫폼

### 1. HuggingFace Spaces (가장 추천 ⭐)

**장점**:
- ✅ 완전 무료
- ✅ 설정 간단
- ✅ Streamlit/FastAPI 모두 지원
- ✅ 모델 호스팅 가능
- ✅ 자동 HTTPS

**단점**:
- CPU만 사용 (느릴 수 있음)
- 디스크 용량 제한 (50GB)

**배포 방법**:
1. https://huggingface.co/spaces 접속
2. "Create new Space" 클릭
3. 설정:
   - SDK: Streamlit
   - Visibility: Public
4. `deployment/huggingface/` 폴더의 파일들을 업로드
5. 자동 배포 완료!

**필요 파일**:
- `app.py` (Streamlit 앱)
- `requirements.txt`
- `README.md`
- 모델 체크포인트 (Git LFS 사용)

### 2. Railway (무료 크레딧)

**장점**:
- ✅ $5 무료 크레딧/월
- ✅ Docker 지원
- ✅ 간단한 설정

**단점**:
- 크레딧 소진 시 유료

### 3. Fly.io (무료 티어)

**장점**:
- ✅ 무료 티어 (3 shared-cpu-1x VMs)
- ✅ 전 세계 엣지 배포
- ✅ 빠른 속도

## 📦 배포 준비

### HuggingFace Spaces 배포

```bash
# 1. HuggingFace 계정 생성 및 로그인
pip install huggingface_hub
huggingface-cli login

# 2. Space 생성
huggingface-cli repo create ai-image-detector --type space

# 3. 파일 업로드
cd deployment/huggingface
huggingface-cli upload ai-image-detector . --repo-type space
```


## 🔧 최적화 팁

### 이미지 크기 줄이기

1. **체크포인트 최적화**: Quantization 사용
2. **불필요한 파일 제외**: .dockerignore 활용
3. **Multi-stage 빌드**: 이미 사용 중

### 성능 향상

1. **모델 경량화**: ONNX 변환
2. **캐싱**: 모델 로드 캐싱 (이미 구현됨)
3. **비동기 처리**: FastAPI 비동기 사용

## 📝 체크리스트

### HuggingFace Spaces
- [ ] HuggingFace 계정 생성
- [ ] Space 생성
- [ ] 파일 업로드
- [ ] 모델 체크포인트 업로드 (Git LFS)
- [ ] 배포 확인


## 🔗 빠른 링크

- [HuggingFace Spaces](https://huggingface.co/spaces)
- [Railway](https://railway.app)
- [Fly.io](https://fly.io)

