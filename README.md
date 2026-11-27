# AdCanvas

음식 사진 한 장과 요청 문장으로 **광고 이미지**를 만드는 생성형 AI 프로젝트입니다.
원본 음식은 그대로 보존하고, 배경만 요청에 맞게 새로 만듭니다.

```
사진 → ① Segmentation(음식 분리) → ② VLM(요청 해석) → ③ 배경 생성 → ④ 원본 음식 재합성 → 광고 이미지 + 광고 문구
```

- 데모 사이트: https://adcanvas-nine.vercel.app/ (서버를 켜 둔 동안 동작)
- 팀: 투투스 (민서연, 신채은, 윤지원, 정지은, 황시은)

## 레포 구성

| 레포 | 내용 |
|---|---|
| **AdCanvas** (이 레포) | 메인 노트북(배포용), 서브 노트북(평가용) |
| [Segmentation-SAM](https://github.com/2026-2-Generative-AI-Project/Segmentation-SAM) | Segmentation: 라벨링, SAM 2.1 파인튜닝, 실험 |
| [VLM](https://github.com/2026-2-Generative-AI-Project/VLM) | 요청 해석: Qwen2.5-VL 파인튜닝 |
| [background-generation](https://github.com/2026-2-Generative-AI-Project/background-generation) | 배경 생성 실험: SDXL·OpenAI 백엔드 비교 (`adgen.py`) |
| [AdCanvas-Frontend](https://github.com/2026-2-Generative-AI-Project/AdCanvas-Frontend) | 웹 화면 (React, Vercel) |

## 노트북

| 파일 | 용도 |
|---|---|
| `AdCanvas_main.ipynb` | **배포용.** 사진 한 장을 끝까지 처리하고, 서버(FastAPI + ngrok)로 웹 화면과 연결 |
| `AdCanvas_eval.ipynb` | **평가용.** 같은 평가 세트로 단계별 비교 (팀별 섹션) |

### 메인 노트북 구성

| 섹션 | 담당 | 서버용 함수 |
|---|---|---|
| 0. 설치 | 공통 | |
| 1. Segmentation | Segmentation팀 | `nukki(image, product_name, boxes)` → `nukki_out` |
| 2. VLM (요청 해석) | VLM팀 | `vlm(nukki_out, form_data)` → `vlm_out` |
| 3. 배경 생성 | 생성팀 | |
| 4. 원본 음식 재합성 | 생성팀 | `generate(nukki_out, vlm_out)` → `final_out` |
| 5. 서버 | Segmentation팀 | `POST /generate`에서 위 세 함수를 차례로 호출 |

### 실행 방법 (서버 켜기)
1. Google Colab에서 `AdCanvas_main.ipynb` 열기 → 런타임 유형 **T4 GPU**
2. **런타임 → 모두 실행**
   - 0번 실행 중 세션이 한 번 다시 시작되면 정상 (설치 중 Pillow 버전이 바뀐 경우). 다시 **모두 실행**
3. 5-2에 나온 주소 뒤에 `/health`를 붙여 열고 `nukki`, `vlm`, `generate`가 모두 `true`인지 확인
4. 끝나면 **런타임 연결 해제 및 삭제** (GPU 사용, 생성 1번마다 OpenAI 요금 발생)

필요한 것
- 공유받은 `AdCanvas_nukki` 폴더를 내 드라이브에 **바로가기로 추가** (가중치, LoRA 어댑터, API 키가 이 경로에서 열림)
- 코랩 보안 비밀 `NGROK_AUTHTOKEN`, `NGROK_DOMAIN` (서버 담당 계정의 토큰과 고정 주소)

### 서비스 구조
```
웹 화면(Vercel) ──POST /generate──▶ ngrok(고정 주소) ──▶ 코랩 GPU (메인 노트북, FastAPI)
```

| 요청 (form-data) | 응답 |
|---|---|
| `image`, `product_name`, `prompt`, `store_name`, `target`, `mood`, `ratio`(1:1, 4:5, 9:16) | `200` `{imageUrl, copy: {headline, body, hashtags}}` |
| `boxes` (선택): 사용자가 그린 박스 `[[x0, y0, x1, y1], ...]`, 원본 사진 픽셀 좌표 | `400` boxes 형식 오류 · `422` 그릇을 못 찾음 (웹 화면이 박스 지정으로 이동) · `500` 서버 오류 · `501` 단계 함수 없음 |

- 서버 주소는 레포에 적지 않음. Vercel 환경 변수와 지라에만 기록

## 단계별 설명

### ① Segmentation (음식 분리)
사진에서 그릇과 음식을 **그릇 하나씩** 분리하고, 광고에 쓸 그릇을 고릅니다.

| 단계 | 모델 | 역할 |
|---|---|---|
| 그릇 찾기 | Grounding DINO tiny | 글자(plate, bowl 등)로 그릇·음식 박스 찾기 |
| 그릇별 마스크 | SAM 2.1 hiera-small + 파인튜닝 디코더 | 박스 하나 → 그릇 + 음식 + 걸친 수저를 통째로 |
| 후처리 | 직접 작성 | 조각 정리, 구멍 채우기, 중복 그릇 제거 |
| 그릇 고르기 (서버) | 직접 작성 | 메인 그릇 자동 선택 (아래) |

- 출력: 음식 마스크(흰색 = 음식, 원본과 같은 크기), 투명 배경 cutout(RGBA)
- 성능 (test 20장, 전체 파이프라인): IoU **0.906**, 경계 BF **0.827** (파인튜닝 전 0.853 / 0.749)
- 자세한 내용: [Segmentation-SAM v1.0](https://github.com/2026-2-Generative-AI-Project/Segmentation-SAM/releases/tag/v1.0)

**그릇 자동 선택 규칙** (서버용 `nukki()`)
1. 그릇마다 점수: 넓을수록, 사진 가운데에 가까울수록 높음. 사진 끝에 잘린 그릇은 크게 낮춤
2. 점수 1등(메인)은 항상 사용
3. 나머지는 메인과 크기가 비슷하고 사진 끝에 안 걸린 그릇만 같이 사용
4. 상품명에 세트·정식·한상·백반·코스 등이 있으면 사진 끝에 안 걸린 그릇 전부 사용

| 사진 | 결과 |
|---|---|
| 메인 하나 + 앞접시·옆 접시 | 메인만 |
| 비슷한 크기의 메인 여러 개 | 다 같이 |
| 메인 + 작은 반찬 | 메인만 |
| 세트·정식 (상품명에 해당 단어) | 사진 끝에 안 걸린 그릇 전부 |

**사용자 박스 보정**: 자동 선택이 틀리거나 그릇을 못 찾으면 사용자가 박스를 그려 다시 생성합니다. 박스 안에 찾아 둔 그릇이 있으면 그 그릇을 쓰고, 없으면 그 영역을 SAM으로 새로 분리합니다.

### ② VLM (요청 해석)
Segmentation 결과(음식 이미지)와 사용자 입력을 함께 분석해, **3단계 배경 생성 프롬프트**와 **한국어 광고 문구**를 구조화된 JSON(`vlm_out`)으로 생성합니다.

| 단계 | 모델 | 역할 |
|---|---|---|
| 전처리 | 없음 (직접 짠 코드) | 투명 배경 cutout(PNG)을 긴 변 512로 줄이고 중립 회색 `RGB(128, 128, 128)` 배경에 합성해 PNG로 저장 (학습과 같은 방식, 압축 손실 없음) |
| 이미지·요청 이해 | Qwen2.5-VL-3B-Instruct + LoRA 파인튜닝 (`merge_and_unload`) | 음식 종류와 사용자 조건을 해석해 배경 묘사·광고 문구·전략 근거를 JSON으로 출력 (`max_pixels=262144`, `do_sample=False`) |
| 후처리 · 예외 방어 | 없음 (직접 짠 코드) | 음식 아래 그림자·자연광 문구와 `duplicate food` 네거티브 보정, 파싱 실패나 필수 값 누락 시 `fallback` 대체 |

- 입력: `nukki_out`(Segmentation 결과) + `form_data`(상품명, 가게명, 요청, 대상 고객, 분위기, 비율)
- 출력 (`vlm_out`): 배경 프롬프트(`sd_prompt`, 영어), 네거티브 프롬프트(`negative_prompt`), 광고 문구(`ad_copy`, 한국어), 전략 설명(`strategy_reason`), 비율(`aspect_ratio`)
- 다음 단계로는 텍스트(`vlm_out`)만 넘어가고, 배경 생성은 1단계의 원본 사진과 마스크를 사용
- 학습 데이터: 음식 사진 100장, 요청 240개 (손·인형이 함께 찍힌 사진, 여러 음식, 빈 입력·무의미한 입력 등 예외 상황 포함)
- 자세한 내용 (학습 데이터 구축 · LoRA 파인튜닝): [VLM](https://github.com/2026-2-Generative-AI-Project/VLM)

**프롬프트 생성 및 후처리 규칙** (서버용 `vlm()`)
1. **음식 중복 묘사 금지**: 4단계에서 원본 음식 픽셀을 그대로 재합성하므로 `sd_prompt`에는 테이블·소품·조명 등 배경만 묘사하고 메인 음식은 기술하지 않음
2. **그림자 · 중복 방지 보정**: 모델 출력에 없으면 `sd_prompt`에 `soft contact shadow beneath the center object, natural directional lighting`(음식 아래 부드러운 그림자, 자연광), `negative_prompt`에 `duplicate food, extra serving of the main dish`를 자동 추가
3. **조건별 규칙 준수**: 빈 입력은 가게명·대상 고객 `미입력`, 분위기 `AI 추천`으로 채움. 가게명이 `미입력`이면 상호 임의 창작 금지, 대상 고객이 `미입력`이면 이미지와 요청으로 추론, 분위기가 `AI 추천`이면 최적 분위기를 선택해 `strategy_reason`에 근거 기재
4. **비율 고정**: `aspect_ratio`는 모델 출력과 관계없이 사용자가 고른 비율(1:1, 4:5, 9:16)로 고정
5. **예외 방어**: JSON 파싱에 실패하거나 필수 3개 키(`sd_prompt`, `ad_copy`, `aspect_ratio`)가 비어 있으면 기본값(`fallback`)으로 대체해 서비스 중단 방지

### ③ 배경 생성 / ④ 원본 음식 재합성
- 배경 생성: OpenAI 이미지 편집 API. 음식 보호 마스크 + 바닥 쪽 그림자 영역을 두고 배경만 생성 (비율별 캔버스)
- 재합성: 생성된 배경 위에 **원본 음식 픽셀을 그대로** 다시 붙임 (원본 해상도 유지)
  - 생성 모델이 음식 위치·크기를 바꿔 그리는 경우가 있어, 생성 결과에서 음식 위치를 특징점 매칭으로 찾아 그 자리에 붙임
  - 사진 끝에 잘린 그릇은 잘린 쪽으로 갈수록 서서히 투명하게, 외곽은 부드럽게 처리
- 평가: 음식 픽셀 보존율, CLIP Score, 생성 시간 (서브 노트북)
- 실험 코드 (SDXL·OpenAI 비교): [background-generation](https://github.com/2026-2-Generative-AI-Project/background-generation)

## 데이터와 보안
- 사진, 정답 마스크, 학습 가중치, API 키는 **드라이브에만** 보관 (레포 Public)
- 서버 주소, ngrok 토큰은 레포에 적지 않음
- 노트북은 출력 없이 커밋

## 사용 모델과 라이선스
| 모델 | 단계 | 라이선스 |
|---|---|---|
| Grounding DINO tiny (IDEA-Research) | Segmentation | Apache-2.0 |
| SAM 2.1 hiera-small (Meta) | Segmentation | Apache-2.0 |
| Qwen2.5-VL-3B-Instruct (Alibaba) | VLM | Qwen Research License (연구·비상업 용도만 허용, 상업 이용은 별도 라이선스 필요) |
| OpenAI 이미지 편집 API | 배경 생성 | OpenAI 이용 약관 |
