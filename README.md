# AdCanvas

음식 사진 한 장과 요청 문장으로 **광고 이미지**를 만드는 생성형 AI 프로젝트입니다.
원본 음식은 그대로 보존하고, 배경만 요청에 맞게 새로 만듭니다.

```
사진 → ① 누끼(음식 분리) → ② VLM(요청 해석) → ③ 배경 생성 → ④ 원본 음식 재합성 → 광고 이미지
```

## 레포 구성

| 레포 | 내용 |
|---|---|
| **AdCanvas** (이 레포) | 메인 노트북(배포용), 서브 노트북(평가용) |
| [Segmentation-SAM](https://github.com/2026-2-Generative-AI-Project/Segmentation-SAM) | 누끼: 라벨링, SAM 2.1 파인튜닝, 실험 |
| [VLM](https://github.com/2026-2-Generative-AI-Project/VLM) | 요청 해석: Qwen2.5-VL 파인튜닝 |
| [AdCanvas-Frontend](https://github.com/2026-2-Generative-AI-Project/AdCanvas-Frontend) | 웹 화면 (React, Vercel) |

## 노트북

| 파일 | 용도 | 상태 |
|---|---|---|
| `AdCanvas_main.ipynb` | **배포용.** 사진 한 장을 끝까지 처리, 서버(FastAPI)로 프론트와 연결 | 누끼 단계 작성 / VLM·생성 단계 작성 예정 |
| (예정) 서브 노트북 | **평가용.** 같은 사진·같은 생성 설정에서 단계별 비교 | 작성 예정 |

### 실행 환경
- Google Colab, GPU(T4) 권장
- 가중치와 데이터는 팀 공유 드라이브에 있음 (레포에 포함하지 않음)
  - 서버를 켜는 사람은 공유받은 `AdCanvas_nukki` 폴더를 내 드라이브에 **바로가기로 추가**해야 같은 경로로 열림

### 서비스 구조 (예정)
```
웹 화면(Vercel) ──POST /generate──▶ ngrok ──▶ 코랩 GPU (메인 노트북, FastAPI)
```

## 단계별 설명

### ① 누끼 (Segmentation) — 황시은, 정지은
사진에서 그릇과 음식을 **그릇 하나씩** 분리합니다.

| 단계 | 모델 | 역할 |
|---|---|---|
| 그릇 찾기 | Grounding DINO tiny | 글자(plate, bowl 등)로 그릇·음식 박스 찾기 |
| 그릇별 마스크 | SAM 2.1 hiera-small + 파인튜닝 디코더 | 박스 하나 → 그릇 + 음식 + 걸친 수저를 통째로 |
| 후처리 | 직접 작성 | 조각 정리, 구멍 채우기, 중복 그릇 제거 |

- 출력: 음식 마스크(흰색 = 음식, 원본과 같은 크기), 투명 배경 누끼(RGBA)
- 성능 (test 20장, 전체 파이프라인): IoU **0.906**, 경계 BF **0.827** (파인튜닝 전 0.853 / 0.749)
- 자세한 내용: [Segmentation-SAM v1.0](https://github.com/2026-2-Generative-AI-Project/Segmentation-SAM/releases/tag/v1.0)

### ② VLM (요청 해석) — 작성 예정

### ③ 배경 생성 / ④ 원본 음식 재합성 — 작성 예정

## 데이터와 보안
- 사진, 정답 마스크, 학습 가중치는 **드라이브에만** 보관 (레포 Public)
- 노트북은 출력 없이 커밋

## 사용 모델과 라이선스
| 모델 | 단계 | 라이선스 |
|---|---|---|
| Grounding DINO tiny (IDEA-Research) | 누끼 | (확인 필요) |
| SAM 2.1 hiera-small (Meta) | 누끼 | (확인 필요) |
| (VLM, 생성 모델) | ②, ③ | 작성 예정 |