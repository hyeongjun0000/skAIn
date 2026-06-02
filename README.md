# skAIn : Real-time AI Skin Analysis System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-3.0.0-black?logo=flask)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8.0-green?logo=opencv)
![ONNX](https://img.shields.io/badge/ONNX_Runtime-1.16.0-blue?logo=onnx)
![License](https://img.shields.io/badge/License-MIT-yellow)

**skAIn PRO**는 웹캠을 통해 사용자의 얼굴을 실시간으로 추적하고, 18가지 피부 지표를 정밀하게 분석하는 지능형 컴퓨터 비전(CV) 웹 애플리케이션입니다. 무거운 딥러닝 모델을 경량화하여 저사양 하드웨어에서도 쾌적하게 구동되도록 최적화되었습니다.

<br>

## ✨ 주요 아키텍처 및 특징 (Key Features)

 **Edge-Optimized AI 추론**: 
  * 원본 딥러닝 모델을 ONNX 형식으로 양자화(Quantization)하여 가중치 용량을 대폭 축소.
  * 맥북(macOS)은 물론, 자원이 제한적인 **라즈베리파이(Raspberry Pi)** 환경에서도 병목 현상 없는 실시간(Real-time) 프레임 처리 보장.
 **지능형 연속 인식 시스템 (Continuous Liveness Detection)**: 
  * 금융권 인증 화면과 동일한 수준의 엄격한 상호작용 UI 적용.
  * 얼굴이 프레임에서 이탈할 경우, 즉각적으로 진행률을 0%로 초기화하여 불량 데이터 분석을 원천 차단.
 **비동기 실시간 통신 (Asynchronous UI)**: 
  * Flask(Backend)와 Vanilla JS(Frontend) 간 200ms 단위의 Fetch API 통신을 통해 새로고침 없는 매끄러운 사용자 경험(UX) 제공.

<br>

## 📊 시스템 동작 순서도 (System Architecture Flow)

```mermaid

graph TD
    %% 노드 스타일 정의
    classDef frontend fill:#f9f9f9,stroke:#333,stroke-width:2px;
    classDef backend fill:#e1f5fe,stroke:#039be5,stroke-width:2px;
    classDef ai fill:#fff3e0,stroke:#f57c00,stroke-width:2px;

🚀 빠른 시작 (Quick Start)
본 프로젝트를 로컬 환경(본인의 컴퓨터)에서 직접 실행해 보는 방법입니다.

1. 레포지토리 클론 (Clone)
Bash
git clone [https://github.com/내아이디/skAIn-PRO.git](https://github.com/내아이디/skAIn.git)
cd skAIn
(주의: 내아이디 부분을 본인의 깃허브 아이디로 변경할 것)

2. 가상환경 세팅 및 의존성 설치
파이썬 환경 충돌을 막기 위해 가상환경(venv 또는 conda) 사용을 권장합니다.

Bash
pip install -r requirements.txt
3. 서버 실행
노트북에 웹캠이 활성화되어 있는지 확인한 후, 아래 명령어를 실행합니다.

Bash
python3 app_web.py
4. 웹 브라우저 접속
터미널에 서버 구동 메시지가 나타나면, 브라우저를 열고 아래 주소로 접속하세요.

👉 http://localhost:5001

* Error 발생 시 *
Mac 환경에서 카메라 불이 들어오지 않거나 에러가 날 때

시스템 설정 > 개인정보 보호 및 보안 > 카메라 메뉴에서 현재 사용 중인 터미널(또는 IDE)의 카메라 접근 권한이 허용되어 있는지 확인해 주세요.

아이폰 '연속성 카메라(Continuity Camera)'가 켜져 있을 경우, app_web.py 내의 cv2.VideoCapture(1)을 0 또는 다른 인덱스로 변경해야 할 수 있습니다.

📝 License
This project is licensed team 19?
