# Docker 설치 가이드 (macOS)

## 🐳 Docker Desktop 설치

### 방법 1: Homebrew 사용 (권장)

터미널에서 다음 명령어 실행:

```bash
brew install --cask docker
```

설치 후 Docker Desktop을 실행:
```bash
open -a Docker
```

### 방법 2: 직접 다운로드

1. https://www.docker.com/products/docker-desktop 접속
2. "Download for Mac" 클릭
3. 다운로드된 `.dmg` 파일 실행
4. Docker.app을 Applications 폴더로 드래그
5. Applications에서 Docker.app 실행

## ✅ 설치 확인

터미널에서 다음 명령어로 확인:

```bash
docker --version
docker-compose --version
```

정상적으로 설치되었다면 버전 정보가 출력됩니다.

## 🚀 Docker Desktop 시작

1. Applications 폴더에서 Docker.app 실행
2. 메뉴바에 Docker 아이콘이 나타날 때까지 대기
3. 아이콘이 초록색이 되면 준비 완료

## 📝 다음 단계

Docker 설치가 완료되면:

```bash
cd deployment/docker
docker-compose up -d
```

자세한 사용법은 `README.md`를 참고하세요.

