# adgen — 광고용 음식 이미지 배경 생성

## 목표

세그멘테이션된 음식 컷아웃에서 **원본 음식 픽셀은 보존하고 배경만 생성**해 목표 비율의 광고 이미지를 만든다.

- 입력: 음식 RGBA 컷아웃(또는 RGB 이미지와 음식 마스크), 배경 프롬프트
- 처리: Inpainting → Outpainting → Paste-back
- 백엔드 비교: `mock`, `sdxl`, `openai`

## 핵심 처리 단계

1. **Inpainting**: 음식 주변의 배경을 생성한다.
2. **Outpainting**: `1:1`, `4:5`, `9:16` 등 목표 비율에 맞게 캔버스를 확장한다.
3. **Paste-back**: 원본 음식 픽셀을 다시 합성한다.

음식 경계에서 `feather_px`보다 안쪽인 픽셀은 원본과 비트 단위로 같아야 한다. `paste_back()`의 확실한 코어와 블러 영역의 합집합 처리를 제거하거나 단순 임계값으로 바꾸지 않는다.

## 개발 환경과 실행

```
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest test_adgen.py -q

# 특정 테스트만 실행
python -m pytest test_adgen.py -q -k "adaptive"

# 로직 확인
python adgen.py --image assets/burger.png --backend mock --ratio 4:5 --prompt "wooden table"
```

```
Copy-Item .env.example .env

# Colab 등 VRAM이 충분한 환경에서 실행 권장
python adgen.py --image assets/burger.png --backend sdxl --ratio 1:1 --prompt "clean marble counter, soft studio light, food photography"

# OpenAI API
python adgen.py --image assets/burger.png --backend openai --ratio 4:5 --prompt "rustic wooden table"
```

## 백엔드

| 백엔드 | 용도 | 주의점 |
|---|---|---|
| `mock` | GPU/API 없이 기하·합성 로직 테스트 | 실제 생성 품질은 검증하지 못함 |
| `sdxl` | SDXL 인페인팅 모델 | VRAM 8GB 이상 권장 |
| `openai` | GPT Image API | 마스크는 권고에 가까워 음식 보존을 보장하지 못함 |

모든 백엔드는 `Backend.generate(image, mask, prompt, cfg, stage) -> Image` 인터페이스를 따른다. 새 백엔드는 `adgen.py`에 `Backend` 하위 클래스와 `BACKENDS` 등록 한 줄을 추가한다. API 키 검사는 객체 생성자가 아니라 `client` 프로퍼티에서 한다.

비교 실험에서는 `run(cfg, backend=...)`에 백엔드 인스턴스를 재사용해 모델을 반복 로드하지 않는다.

## 반드시 지켜야 할 불변식

### 1. 마스크의 흰색 의미를 섞지 않는다

| 이름 | 흰색(255)의 의미 | 용도 |
|---|---|---|
| `food`, `food_mask`, `protect` | 음식 | 음식 보존·paste-back |
| `gen_mask`, `om` | 생성할 영역 | 백엔드 입력 |

`generation_mask()`가 음식 마스크를 생성 마스크로 바꾸는 기준 지점이다. 단, OpenAI는 마스크 극성이 반대다. `mask_rgba()`에서 마스크를 반전해 투명한 영역이 생성 영역이 되게 한다. 이 처리를 제거하면 음식이 다시 그려질 수 있다.

**OpenAI 마스크는 하드 제약이 아니다.** 모델이 음식 자체를 다시 그릴 수 있으므로 반드시 paste-back한다. `fidelity`만으로 모델이 음식을 건드렸는지 알 수 없으니, `04_generated.png`와 `01_flat.png`도 눈으로 비교한다.

### 2. 캔버스·마스크 처리

- 캔버스 크기는 `LATENT`(8)의 배수로 만든다.
- 빈 캔버스는 검정이나 흰색이 아닌 중간 회색(127)으로 채운다.
- 리샘플링: 이진 마스크 `NEAREST`, 소프트 마스크 `BILINEAR`, 이미지 `LANCZOS`.
- `subject_scale`은 캔버스의 가로와 세로 모두를 기준으로 적용한다.
- `fidelity()`는 최종 캔버스의 음식 박스 영역을 `원본.resize(box_size)`와 비교한다. 원본 크기로 되돌려 비교하지 않는다. 보존 테스트에는 단색이 아닌 질감 있는 음식 이미지를 사용한다.

### 3. 경계 정책

- `protect`: 음식 영역을 바깥으로 넓혀 생성에서 제외한다. 후광은 줄지만 접촉 그림자가 부족할 수 있다.
- `blend`: 음식 경계 안쪽을 재생성해 접촉 그림자를 만들 수 있지만 경계 픽셀이 바뀔 수 있다.
- `adaptive`: 음식 하단(`floor_ratio`)은 `blend`, 위쪽은 `protect`로 처리하고 수직 램프로 부드럽게 전환한다. 램프를 없애면 경계에 인공적인 수평선이 생길 수 있다.

### 4. two_stage의 1단계 여백

`two_stage`는 1단계 인페인팅 결과를 2단계 아웃페인팅이 보존한다. 따라서 **1단계 배경이 실패하면 2단계가 고칠 수 없다.**

`stage1_subject_max`(기본값 `0.70`)는 1단계 전에 캔버스에 회색 여백을 추가해 음식이 차지하는 비율을 낮춘다. 음식을 축소하지 않으므로 원본 픽셀은 그대로다. 여백을 너무 많이 추가하면 2단계가 확장할 공간이 줄어든다. 2단계 여백이 부족하면 `subject_scale`을 낮춘다.

- `two_stage`: 인페인팅과 아웃페인팅을 분리해 평가·튜닝할 때 사용한다.
- `oneshot`: 큰 캔버스에서 배경 전체를 한 번에 생성한다. 더 빠르고 조명·원근의 일관성이 좋을 수 있으며, 컷아웃이 매우 타이트할 때 더 안전하다.

`two_stage`의 paste-back과 fidelity는 여백이 추가된 기준 이미지(`ref`, `ref_food`)의 좌표계를 사용한다. 원본 `flat`/`food`를 사용하면 좌표가 어긋난다.

## 확인된 실험 결과

- **SDXL, Colab T4, `burger.png` 500×500**
  - `two_stage`, 768px/24 steps: 약 58초, 대리석 상판·흐린 주방 배경
  - `oneshot`, 768px/24 steps: 약 43초, 대리석 상판·흐린 주방 배경
  - `two_stage`, 1024px/30 steps: 실패. 1단계가 평평한 회색 배경을 생성했고 2단계가 이를 고치지 못함
  - 두 모드 모두 최종 음식 픽셀 보존 지표 `identical_ratio: 1.0`
- **OpenAI, `oneshot`, 1024px, `quality=low`**
  - 약 17초, 배경의 재질·조명 표현이 SDXL보다 풍부했음
  - 모델이 음식을 다시 그렸지만 paste-back으로 원본 음식과 워터마크가 복원됨
  - 최종 음식 픽셀 보존 지표 `identical_ratio: 1.0`


## 테스트와 출력 확인

테스트는 `mock` 백엔드 중심이며, 실제 SDXL·OpenAI 호출 경로와 생성 품질을 모두 검증하지는 않는다. SDK 변경이나 시각적 문제는 별도로 확인한다.

`outputs/<run_name 또는 타임스탬프>/`에서 단계별 PNG와 `manifest.json`을 확인한다. 이음선, 음식 복제, 조명 불일치 등은 단위 테스트만으로 증명할 수 없으므로 결과 이미지를 직접 비교한다. 새 단계를 추가하면 `manifest["stages"]`에도 기록한다.

주요 파일:
- `adgen.py`: 전체 파이프라인과 CLI
- `test_adgen.py`: 모델·GPU·API 키 없이 실행하는 테스트
- `colab_sdxl.ipynb`: Colab T4 SDXL 실행 및 비교 실험
- `requirements.txt`: 의존성
- `assets/burger.png`: 예제 입력