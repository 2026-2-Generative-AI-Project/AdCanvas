# AdCanvas

음식 사진 한 장과 요청 문장으로 **광고 이미지**를 만드는 생성형 AI 프로젝트입니다.
원본 음식은 그대로 보존하고, 배경만 요청에 맞게 새로 만듭니다.

```
사진 → ① Segmentation(음식 분리) → ② VLM(요청 해석) → ③ 배경 생성 → ④ 원본 음식 재합성 → 광고 이미지 + 광고 문구
```

## 진행 상황 (11/25)

| 항목 | 상태 |
|---|---|
| 메인 노트북 (배포용) | 1~5 전부 작성 완료, 서버로 프론트와 연결 |
| 서브 노트북 (평가용) | Segmentation·VLM·생성 평가 작성 완료. 각 팀이 실행해서 보고서·발표 자료에 반영 |
| 웹 화면 ↔ 서버 연결 | 배포 사이트에서 생성까지 확인 |
| 다음 작업 | 사용자 보정(박스로 그릇 지정), 광고 문구 형식 정리, 시연 영상 녹화 |

## 레포 구성

| 레포 | 내용 |
|---|---|
| **AdCanvas** (이 레포) | 메인 노트북(배포용), 서브 노트북(평가용) |
| [Segmentation-SAM](https://github.com/2026-2-Generative-AI-Project/Segmentation-SAM) | Segmentation: 라벨링, SAM 2.1 파인튜닝, 실험 |
| [VLM](https://github.com/2026-2-Generative-AI-Project/VLM) | 요청 해석: Qwen2.5-VL 파인튜닝 |
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
| 1. Segmentation | Segmentation팀 | `nukki(image, product_name)` → `nukki_out` |
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
- 코랩 보안 비밀 `NGROK_AUTHTOKEN` (서버 담당 계정. 고정 주소가 이 계정에 묶여 있음)

### 서비스 구조
```
웹 화면(Vercel) ──POST /generate──▶ ngrok(고정 주소) ──▶ 코랩 GPU (메인 노트북, FastAPI)
```

| 요청 (form-data) | 응답 |
|---|---|
| `image`, `product_name`, `prompt`, `store_name`, `target`, `mood`, `ratio`(1:1, 4:5, 9:16) | `200` `{imageUrl, copy: {headline, body, hashtags}}` |
| | `422` 그릇을 못 찾음 · `500` 서버 오류 · `501` 단계 함수 없음 |

- 서버 주소는 레포에 적지 않음 (공개 레포라 누구나 호출하면 요금 발생). Vercel 환경 변수와 지라에만 기록

## 단계별 설명

### ① Segmentation (누끼 따기)
사진에서 그릇과 음식을 **그릇 하나씩** 분리하고, 광고에 쓸 그릇을 고릅니다.

| 단계 | 모델 | 역할 |
|---|---|---|
| 그릇 찾기 | Grounding DINO tiny | 글자(plate, bowl 등)로 그릇·음식 박스 찾기 |
| 그릇별 마스크 | SAM 2.1 hiera-small + 파인튜닝 디코더 | 박스 하나 → 그릇 + 음식 + 걸친 수저를 통째로 |
| 후처리 | 직접 작성 | 조각 정리, 구멍 채우기, 중복 그릇 제거 |
| 그릇 고르기 (서버) | 직접 작성 | 메인 그릇 자동 선택 (아래) |

- 출력: 음식 마스크(흰색 = 음식, 원본과 같은 크기), 투명 배경 누끼(RGBA)
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

- 예정: 자동 선택이 틀렸을 때 사용자가 **박스로 그릇을 지정**하는 보정 기능

### ② VLM (요청 해석)
- 모델: Qwen2.5-VL-3B-Instruct + LoRA 파인튜닝 어댑터
- 입력: Segmentation 결과(회색 배경) + 사용자 입력(상품명, 가게명, 요청, 대상 고객, 분위기, 비율)
- 출력: 배경 생성 프롬프트(영어), 네거티브 프롬프트, 광고 문구(한국어), 전략 설명, 비율
- 자세한 내용: [VLM](https://github.com/2026-2-Generative-AI-Project/VLM)

### ③ 배경 생성 / ④ 원본 음식 재합성
- 배경 생성: OpenAI 이미지 편집 API. 음식 영역은 보호하고 배경만 생성 (비율별 캔버스)
- 재합성: 생성된 배경 위에 **원본 음식 픽셀을 그대로** 다시 붙임
- 평가: 음식 픽셀 보존율, CLIP Score, 생성 시간 (서브 노트북)

## 데이터와 보안
- 사진, 정답 마스크, 학습 가중치, API 키는 **드라이브에만** 보관 (레포 Public)
- 서버 주소, ngrok 토큰은 레포에 적지 않음
- 노트북은 출력 없이 커밋

## 사용 모델과 라이선스
| 모델 | 단계 | 라이선스 |
|---|---|---|
| Grounding DINO tiny (IDEA-Research) | Segmentation | Apache-2.0 |
| SAM 2.1 hiera-small (Meta) | Segmentation | Apache-2.0 |
| Qwen2.5-VL-3B-Instruct (Alibaba) | VLM | Qwen Research License (확인 필요) |
| OpenAI 이미지 편집 API | 배경 생성 | OpenAI 이용 약관 |
