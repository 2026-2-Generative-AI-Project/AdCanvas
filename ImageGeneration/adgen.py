"""adgen - 음식 사진의 배경만 생성하는 광고 이미지 파이프라인 (단일 파일).

입력: 세그멘테이션된 음식 컷아웃(RGBA) 또는 RGB + 배경 마스크, 배경 프롬프트
출력: 목표 비율에 맞춰 배경이 생성된 이미지. 음식 픽셀은 비트 단위로 보존된다.

세 가지 기법
  Inpainting  음식 영역은 보존하고 배경 마스크 영역만 프롬프트대로 생성
  Outpainting 목표 출력 비율에 맞춰 캔버스를 넓히고 확장 영역을 생성
  Paste-back  생성 후 원본 음식 픽셀을 원해상도로 다시 덮어써 보존을 보장

백엔드
  sdxl    로컬 Stable Diffusion XL 인페인팅 (VRAM 8GB+)
  openai  GPT Image edit API (gpt-image-2.5)
  mock    GPU/API 없이 기하·합성 로직만 검증

사용
  python adgen.py --image assets/burger.png --prompt "marble counter" --ratio 4:5 --backend mock
  from adgen import Config, run
  img, manifest = run(Config(image="assets/burger.png", prompt="...", ratio=(4, 5)))
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import NamedTuple, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

log = logging.getLogger("adgen")

# SDXL VAE 다운샘플 배수. 캔버스 크기는 반드시 이 배수여야 한다.
LATENT = 8

# ============================================================================
# 마스크 규약 - 섞으면 음식과 배경이 뒤바뀐다
#   food_mask : 흰색(255) = 음식 (보존)        <- 알파 채널에서 유도, paste-back에 사용
#   gen_mask  : 흰색(255) = 생성할 영역        <- 백엔드 입력 (diffusers mask_image와 동일)
#   generation_mask()가 전자 -> 후자로 바꾸는 유일한 지점이다.
# ============================================================================


@dataclass
class Config:
    """전체 설정"""

    # --- 입출력 ---
    image: str = "assets/burger.png"
    mask_path: Optional[str] = None      # 배경 마스크(흰색=생성). None이면 알파 채널 사용
    out_dir: str = "outputs"
    run_name: Optional[str] = None       # None이면 타임스탬프
    save_steps: bool = True              # 단계별 PNG 저장 (시각 디버깅용)

    # --- 프롬프트 ---
    prompt: str = "clean marble counter, soft window light, food photography"
    negative_prompt: str = "text, watermark, extra food, duplicated food, people, blurry, cluttered"
    outpaint_prompt: Optional[str] = None  # None이면 prompt 재사용

    # --- 생성 ---
    steps: int = 30
    guidance: float = 7.5
    seed: Optional[int] = 42

    # --- 캔버스 / 아웃페인팅 ---
    ratio: Tuple[int, int] = (1, 1)      # 출력 비율. (4,5) 피드, (9,16) 릴스, (16,9) 배너
    long_side: int = 1024                # 캔버스 긴 변 픽셀
    subject_scale: float = 0.62          # 음식이 캔버스 가로/세로에서 차지할 최대 비율
    anchor: str = "center"               # center bottom bottom-left bottom-right left right
    overlap_px: int = 48                 # 원본 안쪽으로 겹쳐 재생성할 띠 (이음선 제거)

    # --- 경계 정책 ---
    mode: str = "oneshot"              # two_stage | oneshot
    # two_stage 전용: 1단계 인페인팅 프레임에서 음식이 차지할 최대 비율.
    # 음식이 이보다 크면 회색 여백을 덧대 모델에게 장면을 구성할 공간을 준다.
    # 낮출수록 1단계가 안정되지만 2단계(아웃페인팅)가 할 일이 줄어든다.
    stage1_subject_max: float = 0.70
    policy: str = "adaptive"              # protect | blend | adaptive
    protect_px: int = 3                  # protect: 음식 바깥으로 넓혀 생성에서 제외
    blend_px: int = 6                    # blend: 음식 안쪽으로 파고들어 경계 재생성
    floor_ratio: float = 0.20            # adaptive: 음식 하단 몇 비율을 '바닥 접촉'으로 볼지
    feather_px: int = 4                  # 생성 마스크 경계 깃털링
    paste_feather_px: int = 2            # paste-back 경계 깃털링

    # --- 백엔드 ---
    backend: str = "sdxl"                # sdxl | openai | mock
    model: Optional[str] = None          # None이면 백엔드 기본 모델
    quality: str = "auto"                # API 전용: auto | low | medium | high | xhigh | max
    device: str = "auto"                 # sdxl 전용: auto | cuda | cpu | mps
    offload: str = "model"               # sdxl 전용: none | model | sequential


# ============================================================================
# 마스크
# ============================================================================

def _gray(m: Image.Image) -> Image.Image:
    return m if m.mode == "L" else m.convert("L")


def binarize(m: Image.Image, threshold: int = 127) -> Image.Image:
    a = np.asarray(_gray(m), np.uint8)
    return Image.fromarray(np.where(a > threshold, 255, 0).astype(np.uint8), "L")


def _morph(m: Image.Image, radius: int, maximum: bool) -> Image.Image:
    """팽창/침식. PIL 커널을 7x7 이하로 쪼개 반복 적용한다 (큰 커널은 느리다)."""
    if radius <= 0:
        return _gray(m)
    out, left = _gray(m), radius
    while left > 0:
        step = min(left, 3)
        f = ImageFilter.MaxFilter if maximum else ImageFilter.MinFilter
        out = out.filter(f(2 * step + 1))
        left -= step
    return out


def dilate(m: Image.Image, radius: int) -> Image.Image:
    return _morph(m, radius, True)


def erode(m: Image.Image, radius: int) -> Image.Image:
    return _morph(m, radius, False)


def feather(m: Image.Image, radius: int) -> Image.Image:
    """경계를 가우시안으로 흐려 소프트 마스크(0~255 연속값)로 만든다."""
    return _gray(m) if radius <= 0 else _gray(m).filter(ImageFilter.GaussianBlur(radius))


def invert(m: Image.Image) -> Image.Image:
    return Image.fromarray(255 - np.asarray(_gray(m), np.uint8), "L")


def union(a: Image.Image, b: Image.Image) -> Image.Image:
    return Image.fromarray(
        np.maximum(np.asarray(_gray(a), np.uint8), np.asarray(_gray(b), np.uint8)), "L"
    )


def subtract(a: Image.Image, b: Image.Image) -> Image.Image:
    """a AND NOT b."""
    x = np.asarray(_gray(a), np.int16) - np.asarray(_gray(b), np.int16)
    return Image.fromarray(np.clip(x, 0, 255).astype(np.uint8), "L")


def food_bbox(food_mask: Image.Image) -> Tuple[int, int, int, int]:
    a = np.asarray(_gray(food_mask), np.uint8) > 127
    if not a.any():
        return (0, 0, food_mask.width, food_mask.height)
    ys, xs = np.where(a)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def generation_mask(food_mask: Image.Image, cfg: Config) -> Image.Image:
    """음식 마스크(255=음식) -> 생성 마스크(255=생성).

    segmentation 마스크는 음식 경계에서 보통 1~3px 틀린다. 그 오차를 어디로
    흘려보낼지가 품질을 가른다.

    protect  음식을 바깥으로 넓혀 생성에서 제외. 후광(halo) 없음, 대신 접촉 그림자를
             만들 자리가 없어 '붙여넣은 느낌'이 남는다.
    blend    음식 안쪽으로 파고들어 경계를 재생성. 접촉 그림자가 생기지만 경계 픽셀이
             모델에 의해 바뀐다 (paste-back이 되돌림).
    adaptive 음식 bbox 높이의 아래쪽 floor_ratio는 바닥과 닿으므로 blend처럼, 그 위는
             protect처럼 처리한다. 두 마스크를 수직 램프로 선형 보간해 섞는데, 특정
             행에서 그냥 잘라 붙이면 그 행에 계단이 생기고 디퓨전이 그걸 수평선이나
             테이블 모서리 같은 인공물로 그려낸다.
    """
    b = binarize(food_mask)
    if cfg.policy == "protect":
        return feather(invert(dilate(b, cfg.protect_px)), cfg.feather_px)
    if cfg.policy == "blend":
        return feather(invert(erode(b, cfg.blend_px)), cfg.feather_px)
    if cfg.policy != "adaptive":
        raise ValueError(f"알 수 없는 policy: {cfg.policy}")

    _, top, _, bottom = food_bbox(b)
    height = bottom - top
    if height <= 0:
        return feather(invert(b), cfg.feather_px)

    protected = np.asarray(dilate(b, cfg.protect_px), np.float32)   # 위쪽: 음식 보호
    shrunk = np.asarray(erode(b, cfg.blend_px), np.float32)         # 아래쪽: 경계 재생성
    split = bottom - height * max(0.0, min(1.0, cfg.floor_ratio))
    band = max(8.0, cfg.feather_px * 3.0)                           # 전환 구간 폭
    rows = np.arange(b.height, dtype=np.float32)
    w = np.clip((rows - (split - band / 2)) / band, 0, 1)[:, None]   # 0=protect, 1=blend
    blended = np.clip(protected * (1 - w) + shrunk * w, 0, 255).astype(np.uint8)
    return feather(invert(Image.fromarray(blended, "L")), cfg.feather_px)


# ============================================================================
# 캔버스 (아웃페인팅 기하)
# ============================================================================

class Plan(NamedTuple):
    canvas: Tuple[int, int]               # (W, H)
    box: Tuple[int, int, int, int]        # 원본이 놓일 자리 (l, t, r, b)
    scale: float
    clamped: bool                         # subject_scale이 캔버스에 맞춰 줄었는지

    @property
    def box_size(self) -> Tuple[int, int]:
        return (self.box[2] - self.box[0], self.box[3] - self.box[1])


def _snap(v: float) -> int:
    return max(LATENT, int(round(v / LATENT)) * LATENT)


def canvas_size(ratio: Tuple[int, int], long_side: int) -> Tuple[int, int]:
    rw, rh = ratio
    if rw <= 0 or rh <= 0:
        raise ValueError(f"잘못된 비율: {ratio}")
    w, h = (long_side, long_side * rh / rw) if rw >= rh else (long_side * rw / rh, long_side)
    return (_snap(w), _snap(h))


def plan_canvas(src: Tuple[int, int], cfg: Config, bbox=None) -> Plan:
    """원본 크기 + 설정 -> 캔버스 배치."""
    sw, sh = src
    cw, ch = canvas_size(cfg.ratio, cfg.long_side)
    bw = (bbox[2] - bbox[0]) if bbox else sw
    bh = (bbox[3] - bbox[1]) if bbox else sh

    # 음식이 가로·세로 '양쪽 모두' subject_scale 이하를 차지하게 두 축 배율 중 작은 쪽을 쓴다.
    # 긴 변만 기준으로 삼으면 9:16 캔버스에 정사각 컷아웃을 넣을 때 가로가 꽉 차서
    # 좌우로 확장할 여백이 사라진다.
    scale = cfg.subject_scale * min(cw / max(bw, 1), ch / max(bh, 1))
    max_scale = min(cw / sw, ch / sh)
    clamped = scale > max_scale
    scale = min(scale, max_scale)

    pw, ph = max(1, round(sw * scale)), max(1, round(sh * scale))
    cx, cy = (cw - pw) // 2, (ch - ph) // 2
    offsets = {
        "center": (cx, cy), "bottom": (cx, ch - ph), "bottom-left": (0, ch - ph),
        "bottom-right": (cw - pw, ch - ph), "left": (0, cy), "right": (cw - pw, cy),
    }
    if cfg.anchor not in offsets:
        raise ValueError(f"알 수 없는 anchor: {cfg.anchor}")
    x, y = (max(0, v) for v in offsets[cfg.anchor])
    return Plan((cw, ch), (x, y, x + pw, y + ph), scale, clamped)


def place(image: Image.Image, plan: Plan, fill=(127, 127, 127)) -> Image.Image:
    """원본을 캔버스에 배치. 빈 영역은 중간 회색으로 채운다.

    검정/흰색으로 채우면 디퓨전이 그 극단값에 끌려가 어둡거나 밝은 배경을 만든다.
    """
    c = Image.new("RGB", plan.canvas, fill)
    c.paste(image.convert("RGB").resize(plan.box_size, Image.LANCZOS), plan.box[:2])
    return c


def place_mask(mask: Image.Image, plan: Plan) -> Image.Image:
    """마스크를 같은 기하로 배치. 이진성 유지를 위해 NEAREST."""
    c = Image.new("L", plan.canvas, 0)
    c.paste(_gray(mask).resize(plan.box_size, Image.NEAREST), plan.box[:2])
    return c


def outpaint_mask(plan: Plan, cfg: Config, protect: Optional[Image.Image] = None) -> Image.Image:
    """확장 영역 마스크(255=생성).

    box 밖 전체 + box 안쪽 overlap_px 띠. 안쪽 띠가 없으면 원본 경계에 '액자 선'이
    그대로 남는다. protect(255=음식)는 생성에서 제외한다.
    """
    m = Image.new("L", plan.canvas, 255)
    l, t, r, b = plan.box
    keep = (l + cfg.overlap_px, t + cfg.overlap_px, r - cfg.overlap_px, b - cfg.overlap_px)
    if keep[2] <= keep[0] or keep[3] <= keep[1]:   # overlap이 너무 커서 보존 영역이 사라짐
        keep = plan.box
    ImageDraw.Draw(m).rectangle(keep, fill=0)
    if protect is not None:
        m = subtract(m, binarize(protect))
    # 이음선은 음식 경계보다 넓게 섞어야 '액자 선'이 안 보인다
    return feather(m, max(cfg.feather_px, 8))


def pad_for_stage1(flat: Image.Image, food: Image.Image, bbox: Tuple[int, int, int, int],
                   max_subject: float) -> Tuple[Image.Image, Image.Image, Tuple[int, int, int, int]]:
    """1단계 인페인팅 전에 회색 여백을 덧댄다. (여백 추가된 RGB, 마스크, bbox)를 돌려준다.

    왜 필요한가 - 세그멘테이션 컷아웃은 음식에 딱 맞게 잘려 있는 경우가 많다
    (burger.png는 음식이 프레임의 87%). 그 상태로 인페인팅하면 "거대한 피사체 주위의
    얇은 테두리를 채워라"가 되어 모델이 장면을 구성할 공간이 없고 결과 분산이 커진다.
    실제로 같은 seed에서도 해상도가 바뀌면 대리석 주방이 나오기도, 평평한 회색이
    나오기도 했다.

    여백을 덧대 음식이 max_subject 이하를 차지하게 만들면 1단계가 안정된다.
    이미 여백이 충분하면(k <= 1) 아무것도 하지 않는다.

    트레이드오프 - 여백을 많이 줄수록 1단계는 안정되지만, 그만큼 배경을 1단계가
    미리 만들어버려 2단계 아웃페인팅이 할 일이 줄어든다.
    """
    sw, sh = flat.size
    bw, bh = max(bbox[2] - bbox[0], 1), max(bbox[3] - bbox[1], 1)
    target = max(0.05, min(1.0, max_subject))
    k = max(bw / (sw * target), bh / (sh * target), 1.0)
    if k <= 1.0 + 1e-6:
        return flat, _gray(food), bbox

    pw, ph = _snap(sw * k), _snap(sh * k)
    off = ((pw - sw) // 2, (ph - sh) // 2)
    padded = Image.new("RGB", (pw, ph), (127, 127, 127))
    padded.paste(flat.convert("RGB"), off)
    mask = Image.new("L", (pw, ph), 0)
    mask.paste(_gray(food), off)
    moved = (bbox[0] + off[0], bbox[1] + off[1], bbox[2] + off[0], bbox[3] + off[1])
    return padded, mask, moved


# ============================================================================
# 합성 (픽셀 보존)
# ============================================================================

def flatten(rgba: Image.Image, fill=(127, 127, 127)) -> Image.Image:
    """RGBA 컷아웃을 중립 회색 위에 올려 RGB로. 투명을 검정으로 두면 생성이 어두워진다."""
    im = rgba.convert("RGBA")
    base = Image.new("RGB", im.size, fill)
    base.paste(im, (0, 0), im.getchannel("A"))
    return base


def paste_back(generated: Image.Image, original: Image.Image, food_mask: Image.Image,
               feather_px: int = 2) -> Image.Image:
    """생성 결과 위에 원본 음식 픽셀을 덮어쓴다. 세 이미지는 같은 크기여야 한다.

    왜 필요한가 - 디퓨전 인페인팅은 "마스크 밖 보존"이라고 하지만 실제로는 이미지
    전체를 VAE로 인코딩/디코딩해 색이 밀리고 질감이 뭉개진다. API 백엔드는 아예
    이미지 전체를 새로 생성한다. 광고용 제품 픽셀엔 허용되지 않는다.

    보장 - 음식 경계에서 feather_px보다 안쪽인 픽셀은 원본과 비트 단위로 동일하다.
    가우시안 블러는 무한히 퍼지므로 블러 결과만 쓰면 코어에서도 마스크가 254가 되어
    생성 배경색이 1/255 섞인다. 그래서 '확실한 코어'(feather_px만큼 침식한 마스크)를
    블러 결과와 합집합으로 묶어 코어를 255로 고정한다. 임계값 튜닝에 의존하지 않고
    성립하는 구조적 보장이므로, 이 합집합을 제거하는 리팩터링은 금지다.
    """
    if not (generated.size == original.size == food_mask.size):
        raise ValueError(f"크기 불일치: {generated.size} {original.size} {food_mask.size}")
    gen, org = generated.convert("RGB"), original.convert("RGB")
    b = binarize(food_mask)
    if feather_px <= 0:
        return Image.composite(org, gen, b)
    # 안쪽으로 살짝 침식한 뒤 깃털링하면 블렌딩 구간이 음식 바깥이 아니라
    # 경계 위에 걸려 후광이 생기지 않는다.
    blurred = feather(erode(b, max(1, feather_px // 2)), feather_px)
    return Image.composite(org, gen, union(blurred, erode(b, feather_px)))


def paste_back_box(canvas: Image.Image, original: Image.Image, food_mask: Image.Image,
                   box: Tuple[int, int, int, int], feather_px: int = 2) -> Image.Image:
    """캔버스 좌표계에서의 paste-back. box 영역에만 원본 음식을 덮어쓴다."""
    size = (box[2] - box[0], box[3] - box[1])
    org = original.convert("RGB").resize(size, Image.LANCZOS)
    fm = binarize(food_mask).resize(size, Image.LANCZOS)
    out = canvas.convert("RGB").copy()
    out.paste(paste_back(out.crop(box), org, fm, feather_px), box[:2])
    return out


def fidelity(result: Image.Image, original: Image.Image, food_mask: Image.Image,
             box: Tuple[int, int, int, int], feather_px: int = 2) -> dict:
    """음식 코어가 보존됐는지 측정. 백엔드를 바꿔가며 비교하는 공정한 기준.

    측정은 반드시 **캔버스 좌표계**에서 한다. 결과를 원본 크기로 되돌려 비교하면
    LANCZOS 확대->축소 왕복이 끼어들어, 음식이 멀쩡한데도 질감 있는 사진에서는
    값이 크게 떨어진다 (보존이 깨진 게 아니라 측정이 틀린 것).
    """
    size = (box[2] - box[0], box[3] - box[1])
    ref = original.convert("RGB").resize(size, Image.LANCZOS)      # paste-back이 붙인 기준
    fm = binarize(food_mask).resize(size, Image.LANCZOS)
    core = np.asarray(erode(fm, max(feather_px, 1)), np.uint8) > 127
    if not core.any():
        return {"core_px": 0, "mean_abs_diff": 0.0, "max_abs_diff": 0, "identical_ratio": 1.0}
    a = np.asarray(ref, np.int16)[core]
    b = np.asarray(result.convert("RGB").crop(box), np.int16)[core]
    d = np.abs(a - b)
    return {
        "core_px": int(core.sum()),
        "mean_abs_diff": round(float(d.mean()), 4),
        "max_abs_diff": int(d.max()),
        "identical_ratio": round(float((d.max(axis=-1) == 0).mean()), 4),
    }


# ============================================================================
# 백엔드 - 인페인팅과 아웃페인팅은 같은 연산이다: (이미지, 마스크, 프롬프트) -> 이미지
#   그래서 메서드가 하나뿐이고, 새 백엔드를 붙여도 run()은 바뀌지 않는다.
#   반환 크기는 run()이 보정하므로 백엔드는 신경 쓰지 않아도 된다.
# ============================================================================

class Backend:
    name = "base"
    last_usage = None        # API 백엔드가 토큰/비용 정보를 남기는 자리

    def generate(self, image: Image.Image, mask: Image.Image, prompt: str,
                 cfg: Config, stage: str) -> Image.Image:
        raise NotImplementedError


class MockBackend(Backend):
    """GPU/API 없이 기하·합성 로직을 검증하는 더미. 실제 백엔드와 같은 경로를 탄다."""
    name = "mock"

    def generate(self, image, mask, prompt, cfg, stage):
        w, h = image.size
        d = hashlib.sha256(f"{prompt}|{cfg.seed}|{stage}".encode()).digest()
        # 주의: bytes를 슬라이스하면 bytes가 나온다(np.array가 못 받는다). frombuffer로 푼다.
        c0, c1 = (np.frombuffer(d[i:i + 3], np.uint8).astype(np.float32) for i in (0, 3))
        gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
        t = ((gx / max(w - 1, 1)) + (gy / max(h - 1, 1))) / 2
        bg = c0 * (1 - t[..., None]) + c1 * t[..., None]
        bg += np.random.default_rng(d[6]).normal(0, 4, bg.shape)
        gen = Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8), "RGB")
        return Image.composite(gen, image.convert("RGB"), _gray(mask))


class SDXLBackend(Backend):
    """로컬 Stable Diffusion XL 인페인팅 (diffusers).

    기본 모델은 SDXL base에 9채널 입력(latent 4 + masked latent 4 + mask 1) UNet을
    붙여 인페인팅 전용으로 파인튜닝한 공식 변형이다. 일반 SDXL base로도 인페인팅은
    되지만(마스크 블렌딩 방식) 경계 일관성이 눈에 띄게 떨어진다.
    """
    name = "sdxl"
    DEFAULT_MODEL = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
    MIN_VRAM_GB = 8.0

    def __init__(self, cfg: Config):
        self.model = cfg.model or self.DEFAULT_MODEL
        self.offload = cfg.offload
        self.device = cfg.device
        self.pipe = None

    def _load(self):
        if self.pipe is not None:
            return
        import torch
        from diffusers import AutoPipelineForInpainting

        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # CPU에서 fp16은 커널이 없어 느리거나 실패한다
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        if self.device == "cuda":
            vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            if vram < self.MIN_VRAM_GB:
                log.warning(
                    "VRAM %.1fGB는 SDXL 인페인팅(권장 %.0fGB)에 부족합니다. sequential 오프로딩으로 "
                    "전환하지만 매우 느립니다 - Colab이나 --backend mock을 권합니다.",
                    vram, self.MIN_VRAM_GB)
                self.offload = "sequential"

        log.info("SDXL 로딩: %s (device=%s, dtype=%s)", self.model, self.device, dtype)
        kw = dict(torch_dtype=dtype, use_safetensors=True)
        try:
            # fp16 variant가 있으면 다운로드 용량이 절반이다
            self.pipe = AutoPipelineForInpainting.from_pretrained(
                self.model, variant="fp16" if dtype == torch.float16 else None, **kw)
        except Exception as e:
            log.info("fp16 variant 없음(%s) - 기본 가중치로 재시도", type(e).__name__)
            self.pipe = AutoPipelineForInpainting.from_pretrained(self.model, **kw)

        # 아래로 갈수록 VRAM을 덜 쓰고 더 느리다. 메모리가 충분하면 none이 가장 빠르다.
        if self.offload == "sequential":
            self.pipe.enable_sequential_cpu_offload()   # 레이어 단위 CPU<->GPU 이동
        elif self.offload == "model":
            self.pipe.enable_model_cpu_offload()        # 모듈 단위 이동
        else:
            self.pipe.to(self.device)
        if self.offload != "none":
            self.pipe.enable_vae_slicing()
            self.pipe.enable_vae_tiling()

    def generate(self, image, mask, prompt, cfg, stage):
        import torch
        self._load()
        # SDXL은 긴 변 1024 근방 + 8의 배수에서 가장 안정적이다. 그 밖에서는 구도가
        # 깨지거나 같은 물체가 반복 생성된다.
        w, h = image.size
        s = cfg.long_side / max(w, h)
        size = (_snap(w * s), _snap(h * s))
        gen = None
        if cfg.seed is not None:
            # 오프로딩 중에는 파이프라인 device가 바뀌므로 CPU 제너레이터가 안전하다
            gen = torch.Generator(device="cpu").manual_seed(cfg.seed)
        return self.pipe(
            prompt=prompt,
            negative_prompt=cfg.negative_prompt or None,
            image=image.resize(size, Image.LANCZOS),
            # 마스크는 BILINEAR - 소프트 엣지를 보존해야 경계가 매끄럽다
            mask_image=_gray(mask).resize(size, Image.BILINEAR),
            width=size[0], height=size[1],
            num_inference_steps=cfg.steps,
            guidance_scale=cfg.guidance,
            generator=gen,
        ).images[0]


class OpenAIBackend(Backend):
    """GPT Image edit API (gpt-image-2.5).

    모델 선택
      sunburst  편집 정밀도가 가장 중요한 워크플로용 - 이 프로젝트의 기본값
      flare     빠르고 고품질인 일상 생성용. --model gpt-image-2.5-flare 로 교체

    SDXL과 다른 세 가지
      1) 마스크 규약이 반대다. **알파가 0(투명)인 곳**이 재생성 영역이다. 그래서
         gen_mask(255=생성)를 invert해 알파 채널에 넣는다. 이걸 뒤집으면 배경이 아니라
         음식이 다시 그려진다.
      2) 마스크가 하드 제약이 아니다. DALL-E 2와 달리 GPT Image는 마스크를 '권고'로
         쓰기 때문에 모양을 정확히 따르지 않을 수 있다. 음식 보존은 마스크가 아니라
         paste-back이 책임진다.
      3) seed와 네거티브 프롬프트가 없다. 재현이 불가능하므로 비교 실험에서는
         같은 프롬프트로 여러 장 뽑아 평균적인 경향을 봐야 한다.

    크기는 16의 배수 / 종횡비 1:3~3:1 / 최대 3840px 범위에서 자유롭다. 범위를 벗어나면
    권장 크기(1024x1024, 1536x1024, 1024x1536) 중 비율이 가장 가까운 쪽으로 보낸다.
    """
    name = "openai"
    DEFAULT_MODEL = "gpt-image-2.5-sunburst"
    RECOMMENDED = [(1024, 1024), (1536, 1024), (1024, 1536)]
    STEP = 16        # 폭/높이는 16의 배수여야 한다
    MAX_EDGE = 3840

    def __init__(self, cfg: Config):
        self.model = cfg.model or self.DEFAULT_MODEL
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI   # pip install openai
            if not os.environ.get("OPENAI_API_KEY"):
                raise RuntimeError("OPENAI_API_KEY를 .env 파일에 설정해야 합니다")
            self._client = OpenAI()
        return self._client

    @classmethod
    def api_size(cls, w: int, h: int) -> Tuple[int, int]:
        """캔버스 크기 -> API가 받는 크기. 비율을 유지한 채 16의 배수로 스냅한다."""
        ar = w / h
        if not (1 / 3 <= ar <= 3) or max(w, h) > cls.MAX_EDGE:
            return min(cls.RECOMMENDED, key=lambda s: abs(s[0] / s[1] - ar))
        snap = lambda v: max(cls.STEP, int(round(v / cls.STEP)) * cls.STEP)
        return (snap(w), snap(h))

    @staticmethod
    def _png(image: Image.Image, name: str) -> io.BytesIO:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        buf.name = name          # SDK가 파일 이름으로 MIME을 판단한다
        buf.seek(0)
        return buf

    @staticmethod
    def mask_rgba(rgb: Image.Image, gen_mask: Image.Image) -> Image.Image:
        """gen_mask(255=생성) -> 알파가 0인 곳이 재생성되는 RGBA 마스크.

        API 호출 없이 극성을 검증할 수 있도록 분리했다. 이 invert를 빼먹으면
        배경이 아니라 음식이 다시 그려진다.
        """
        out = rgb.convert("RGBA")
        out.putalpha(invert(_gray(gen_mask)).resize(rgb.size, Image.BILINEAR))
        return out

    def generate(self, image, mask, prompt, cfg, stage):
        size = self.api_size(*image.size)
        rgb = image.convert("RGB").resize(size, Image.LANCZOS)

        # image는 완전 불투명 RGBA, mask는 같은 RGB에 알파만 바꾼 RGBA.
        # 둘 다 PNG/RGBA/동일 크기여야 한다 ("같은 포맷과 크기" 요구).
        img_rgba = rgb.convert("RGBA")
        mask_rgba = self.mask_rgba(rgb, mask)

        text = prompt
        if cfg.negative_prompt:   # 네거티브 프롬프트가 없으므로 본문에 녹인다
            text = f"{prompt}. Avoid: {cfg.negative_prompt}."
        log.info("[%s] openai %s %dx%d quality=%s", stage, self.model, *size, cfg.quality)
        r = self.client.images.edit(
            model=self.model, prompt=text,
            image=self._png(img_rgba, "image.png"),
            mask=self._png(mask_rgba, "mask.png"),
            size=f"{size[0]}x{size[1]}",
            quality=cfg.quality,
        )
        # 토큰 사용량을 manifest에 남겨 비용을 SDXL의 '시간'과 함께 비교할 수 있게 한다
        usage = getattr(r, "usage", None)
        self.last_usage = usage.model_dump() if hasattr(usage, "model_dump") else usage
        if self.last_usage:
            log.info("[%s] openai usage=%s", stage, self.last_usage)

        if not r.data or not getattr(r.data[0], "b64_json", None):
            raise RuntimeError(
                f"응답에 이미지가 없습니다. model={self.model} "
                f"(gpt-image 계열은 조직 인증이 필요할 수 있습니다): {r}")
        return Image.open(io.BytesIO(base64.b64decode(r.data[0].b64_json)))


BACKENDS = {"mock": MockBackend, "sdxl": SDXLBackend, "openai": OpenAIBackend}


def make_backend(cfg: Config) -> Backend:
    if cfg.backend not in BACKENDS:
        raise ValueError(f"알 수 없는 백엔드 '{cfg.backend}'. 사용 가능: {sorted(BACKENDS)}")
    klass = BACKENDS[cfg.backend]
    return klass() if klass is MockBackend else klass(cfg)


# ============================================================================
# 파이프라인
# ============================================================================

def load_inputs(cfg: Config) -> Tuple[Image.Image, Image.Image]:
    """(원본 RGBA, 음식 마스크 255=음식)를 돌려준다."""
    path = Path(cfg.image)
    if not path.exists():
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {path}")
    src = Image.open(path)
    src.load()

    if cfg.mask_path:
        bg = Image.open(cfg.mask_path).convert("L")
        if bg.size != src.size:
            log.warning("마스크 크기(%s)를 이미지(%s)에 맞춥니다", bg.size, src.size)
            bg = bg.resize(src.size, Image.NEAREST)
        food = invert(binarize(bg))          # 배경 마스크(흰색=배경) -> 음식 마스크
        rgba = src.convert("RGBA")
        rgba.putalpha(food)
        return rgba, food

    if src.mode not in ("RGBA", "LA", "P"):
        raise ValueError(
            f"알파 채널이 없는 이미지({src.mode})입니다. RGBA 세그멘테이션 결과를 쓰거나 "
            "--mask로 배경 마스크를 주세요.")
    rgba = src.convert("RGBA")
    return rgba, binarize(rgba.getchannel("A"))


def run(cfg: Config, backend: Optional[Backend] = None) -> Tuple[Image.Image, dict]:
    """파이프라인 실행. (최종 이미지, manifest)를 돌려준다.

    two_stage 원본 해상도에서 배경 인페인팅 -> 캔버스 확장 아웃페인팅.
                두 기법을 분리해 각각 평가/튜닝할 수 있다.
                주의: 2단계는 1단계 결과를 **고치지 않는다**. 아웃페인팅 마스크가
                붙여넣은 박스 내부를 보존하므로, 1단계가 실패하면 전체가 실패한다.
                컷아웃에 여백이 거의 없으면(음식이 프레임의 80% 이상) 1단계는
                '얇은 테두리 채우기'가 되어 결과 분산이 커진다. 그럴 땐 oneshot을 쓴다.
    oneshot   캔버스에 올린 뒤 배경 전체를 한 번에 생성. 모델에게 장면을 구성할
                공간이 충분하고, 조명 방향·원근의 전역 일관성이 더 좋으며 호출이 1회다.

    backend를 넘기면 그 인스턴스를 재사용한다. 설정만 바꿔가며 여러 번 돌리는
    비교 실험에서는 반드시 넘길 것 - 넘기지 않으면 호출마다 모델을 새로 로드한다
    (SDXL 기준 매번 10~15초 낭비).

        backend = make_backend(cfg)
        for policy in ["protect", "blend", "adaptive"]:
            img, man = run(replace(cfg, policy=policy), backend=backend)
    """
    t0 = time.time()
    out = Path(cfg.out_dir) / (cfg.run_name or time.strftime("%Y%m%d-%H%M%S"))
    out.mkdir(parents=True, exist_ok=True)

    rgba, food = load_inputs(cfg)
    flat = flatten(rgba)
    backend = backend or make_backend(cfg)
    bbox = food_bbox(food)
    log.info("입력 %s, 음식 bbox=%s, 백엔드=%s, mode=%s", rgba.size, bbox, backend.name, cfg.mode)

    man = {"config": asdict(cfg), "backend": backend.name, "input_size": rgba.size,
           "food_bbox": bbox, "stages": []}

    def save(im, name):
        if cfg.save_steps:
            im.save(out / name)

    def gen(image, mask, prompt, stage):
        t = time.time()
        r = backend.generate(image, mask, prompt, cfg, stage)
        if r.size != image.size:     # 백엔드 크기 계약을 여기서 한 번에 보정
            r = r.convert("RGB").resize(image.size, Image.LANCZOS)
        entry = {"stage": stage, "sec": round(time.time() - t, 2)}
        if getattr(backend, "last_usage", None):
            entry["usage"] = backend.last_usage
        man["stages"].append(entry)
        return r

    save(food, "00_food_mask.png")
    save(flat, "01_flat.png")

    def make_plan(size, box):
        p = plan_canvas(size, cfg, box)
        if p.clamped:
            log.warning("subject_scale이 캔버스에 맞춰 축소됐습니다(배율 %.3f). 확장 여백을 "
                        "늘리려면 subject_scale을 낮추세요.", p.scale)
        return p

    if cfg.mode == "two_stage":
        # 1단계에 공간을 준다. 컷아웃이 음식에 딱 맞게 잘려 있으면 인페인팅이
        # '얇은 테두리 채우기'가 되어 결과가 불안정하다 (pad_for_stage1 docstring 참고).
        ref, ref_food, ref_bbox = pad_for_stage1(flat, food, bbox, cfg.stage1_subject_max)
        if ref.size != flat.size:
            before = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / max(flat.size)
            after = max(ref_bbox[2] - ref_bbox[0], ref_bbox[3] - ref_bbox[1]) / max(ref.size)
            log.info("1단계 여백 추가: %s -> %s (음식 %.0f%% -> %.0f%%)",
                     flat.size, ref.size, before * 100, after * 100)
        man["stage1_canvas"] = {"size": ref.size, "padded": ref.size != flat.size}
        plan = make_plan(ref.size, ref_bbox)

        gm = generation_mask(ref_food, cfg)
        save(ref, "01b_stage1_canvas.png")
        save(gm, "02_inpaint_mask.png")
        inp = paste_back(gen(ref, gm, cfg.prompt, "inpaint"), ref, ref_food, cfg.paste_feather_px)
        save(inp, "03_inpainted.png")

        base = place(inp, plan)
        # 음식은 어떤 경우에도 아웃페인팅 대상이 아니다 - 여유를 두고 보호
        protect = dilate(place_mask(ref_food, plan), max(cfg.protect_px, 2))
        om = outpaint_mask(plan, cfg, protect)
        save(base, "04_canvas.png")
        save(om, "05_outpaint_mask.png")
        raw = gen(base, om, cfg.outpaint_prompt or cfg.prompt, "outpaint")
        save(raw, "06_outpainted.png")
    elif cfg.mode == "oneshot":
        ref, ref_food = flat, food
        plan = make_plan(rgba.size, bbox)
        base = place(flat, plan)
        gm = generation_mask(place_mask(food, plan), cfg)
        save(base, "02_canvas.png")
        save(gm, "03_gen_mask.png")
        raw = gen(base, gm, cfg.prompt, "oneshot")
        save(raw, "04_generated.png")
    else:
        raise ValueError(f"알 수 없는 mode: {cfg.mode}")

    # paste-back / fidelity는 plan.box가 가리키는 이미지(two_stage면 여백 포함본)를 쓴다
    final = paste_back_box(raw, ref, ref_food, plan.box, cfg.paste_feather_px)
    man["canvas"] = {"size": plan.canvas, "box": plan.box,
                     "scale": round(plan.scale, 4), "clamped": plan.clamped}
    man["fidelity"] = fidelity(final, ref, ref_food, plan.box, cfg.paste_feather_px)
    man["elapsed_sec"] = round(time.time() - t0, 2)

    final.save(out / "99_final.png")
    (out / "manifest.json").write_text(
        json.dumps(man, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    log.info("완료 (%.1fs) -> %s", man["elapsed_sec"], out / "99_final.png")
    log.info("음식 픽셀 보존: %s", man["fidelity"])
    return final, man


# ============================================================================
# CLI
# ============================================================================

def _ratio(text: str) -> Tuple[int, int]:
    sep = ":" if ":" in text else "x"
    try:
        w, h = text.split(sep)
        return (int(w), int(h))
    except ValueError:
        raise argparse.ArgumentTypeError(f"비율은 'W:H' 형식이어야 합니다 (받은 값: {text})")


def main(argv=None) -> int:
    d = Config()
    p = argparse.ArgumentParser(
        description="세그멘테이션된 음식 이미지 + 배경 프롬프트 -> 광고용 배경 생성")
    p.add_argument("--image", default=d.image)
    p.add_argument("--mask", dest="mask_path", help="배경 마스크(흰색=생성). 없으면 알파 채널 사용")
    p.add_argument("--prompt", default=d.prompt)
    p.add_argument("--negative", dest="negative_prompt", default=d.negative_prompt)
    p.add_argument("--outpaint-prompt", help="아웃페인팅 전용 프롬프트 (기본: --prompt 재사용)")
    p.add_argument("--ratio", type=_ratio, default=d.ratio, help="출력 비율: 1:1, 4:5, 9:16 ...")
    p.add_argument("--long-side", type=int, default=d.long_side)
    p.add_argument("--subject-scale", type=float, default=d.subject_scale)
    p.add_argument("--anchor", default=d.anchor,
                   choices=["center", "bottom", "bottom-left", "bottom-right", "left", "right"])
    p.add_argument("--mode", default=d.mode, choices=["two_stage", "oneshot"])
    p.add_argument("--stage1-subject-max", type=float, default=d.stage1_subject_max,
                   help="two_stage 전용: 1단계 프레임에서 음식이 차지할 최대 비율 "
                        "(초과하면 회색 여백을 덧대 1단계를 안정화)")
    p.add_argument("--policy", default=d.policy, choices=["protect", "blend", "adaptive"],
                   help="음식/배경 경계 처리 정책")
    p.add_argument("--floor-ratio", type=float, default=d.floor_ratio,
                   help="adaptive 전용: 음식 하단 몇 비율을 바닥 접촉으로 볼지")
    p.add_argument("--feather", type=int, dest="feather_px", default=d.feather_px)
    p.add_argument("--overlap", type=int, dest="overlap_px", default=d.overlap_px)
    p.add_argument("--steps", type=int, default=d.steps)
    p.add_argument("--guidance", type=float, default=d.guidance)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--backend", default=d.backend, choices=sorted(BACKENDS))
    p.add_argument("--model", help="백엔드 기본 모델 대신 쓸 모델 id "
                                   "(예: gpt-image-2.5-flare)")
    p.add_argument("--quality", default=d.quality,
                   choices=["auto", "low", "medium", "high", "xhigh", "max"],
                   help="API 백엔드 전용 품질 설정")
    p.add_argument("--device", default=d.device, choices=["auto", "cuda", "cpu", "mps"])
    p.add_argument("--offload", default=d.offload, choices=["none", "model", "sequential"])
    p.add_argument("--out-dir", default=d.out_dir)
    p.add_argument("--run-name")
    p.add_argument("--no-steps", action="store_true", help="단계별 PNG 저장 생략")
    p.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
    cfg = Config(**{k: v for k, v in vars(a).items()
                    if k not in ("verbose", "no_steps") and v is not None})
    cfg.save_steps = not a.no_steps
    run(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
