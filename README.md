# ComfyUI_CustomNode_Image2Spectrum

ComfyUI 커스텀 노드로, 이미지의 주파수 스펙트럼 시각화를 위한 도구입니다.

## 기능

이 노드는 입력 이미지에 2D 푸리에 변환(FFT)을 적용하여 진폭 스펙트럼을 시각화합니다. 영상 처리, 이미지 분석 또는 예술적 효과를 위해 사용할 수 있습니다.

### 주요 특징

- 이미지를 주파수 도메인으로 변환
- 주파수 스펙트럼의 시각화 강도 조절 기능
- 그레이스케일 변환 자동 적용
- 로그 스케일을 사용한 진폭 스펙트럼 시각화

## 설치 방법

1. 이 저장소를 ComfyUI의 custom_nodes 디렉토리에 클론합니다:
   ```
   cd ComfyUI/custom_nodes
   git clone https://github.com/bemoregt/ComfyUI_CustomNode_Image2Spectrum.git
   ```

2. 필요한 의존성 패키지가 설치되어 있는지 확인합니다:
   - PIL (Pillow)
   - numpy
   - torch
   - scipy

## 사용 방법

ComfyUI 워크플로우에서:

1. 노드 추가 메뉴에서 "Image > Image_Spectrum" 노드를 선택합니다.
2. 변환하려는 이미지를 연결합니다.
3. 진폭 파라미터를 조정하여 스펙트럼 시각화의 강도를 조절합니다 (기본값: 20, 범위: 1-50).

## 입력 및 출력

### 입력
- **image**: 변환할 이미지
- **amplitude**: 스펙트럼 시각화의 강도 (기본값: 20, 범위: 1-50)

### 출력
- **IMAGE**: 주파수 스펙트럼 시각화 결과

## 작동 원리

1. 입력 이미지를 그레이스케일로 변환
2. 2D 푸리에 변환(FFT)을 적용
3. 주파수 성분을 중앙으로 이동(fftshift)
4. 주파수 진폭의 로그 스케일 값을 계산하여 시각화
5. 결과를 정규화하여 출력

## 라이선스

MIT 라이선스

## 제작자

- [bemoregt](https://github.com/bemoregt)
