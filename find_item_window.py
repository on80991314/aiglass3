# find_item_window.py
# -*- coding: utf-8 -*-
"""
方案 A：獨立尋物視窗 — 移植自 aiglass2/edge/find_grab_main.py
整合 aiglass3 的 TTS (play_voice_text) 與 ESP32 WebSocket 影像來源。

來源選項（--source）：
  webcam   直接用本機 Webcam (cv2.VideoCapture)，預設
  ws       接收 aiglass3 app_main.py 的 /ws/viewer 廣播影像

TTS 選項（--tts）：
  local    呼叫 aiglass3 的 play_voice_text()（須與 app_main.py 共用環境）
  print    只印到 terminal，不播音（預設，不依賴其他模組）

快捷鍵（視窗內按）：
  q / ESC  關閉
  r        重置狀態機
  c        確認「我拿到了」
  t        尋找「杯子 / cup」（測試用）
  1        水壺 / bottle
  2        手機 / cell phone
  3        筆電 / laptop
  4        碗   / bowl
  5        書   / book
  6        鍵盤 / keyboard
  7        滑鼠 / mouse

啟動範例：
  # 使用本機 Webcam，純 terminal 語音提示
  python find_item_window.py

  # 使用本機 Webcam + aiglass3 TTS（需與 app_main.py 共用 Python 環境）
  python find_item_window.py --tts local

  # 接收 aiglass3 app_main.py 廣播的 ESP32 影像
  python find_item_window.py --source ws --ws-url ws://127.0.0.1:8765/ws/viewer --tts local

  # 接收 ESP32 影像 + PC 麥克風語音辨識
  python find_item_window.py --source ws --ws-url ws://127.0.0.1:8765/ws/viewer --tts local --mic
"""
from __future__ import annotations

# ── 環境變數：防止多執行緒 MKL / OMP 衝突（必須在 import torch / cv2 前設定）
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import ssl
ssl.create_default_context()          # 預先載入 SSL C-library，防止執行緒中途衝突

import faulthandler
faulthandler.enable()

import torch
torch.set_num_threads(1)
torch.set_grad_enabled(False)

import cv2
cv2.setNumThreads(1)

import argparse
import asyncio
import logging
import sys
import threading
import time
import queue
from pathlib import Path
from typing import Callable, List, Optional
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
from PIL import Image, ImageDraw, ImageFont

log = logging.getLogger("find_item_window")


# ════════════════════════════════════════════════════════
#  Detection / HandResult 資料類別
# ════════════════════════════════════════════════════════
@dataclass
class Detection:
    label: str
    conf:  float
    x1: int; y1: int; x2: int; y2: int

    @property
    def cx(self) -> int: return (self.x1 + self.x2) // 2
    @property
    def cy(self) -> int: return (self.y1 + self.y2) // 2
    @property
    def w(self)  -> int: return self.x2 - self.x1
    @property
    def h(self)  -> int: return self.y2 - self.y1


@dataclass
class HandResult:
    tip_x: float; tip_y: float
    x1: float; y1: float; x2: float; y2: float

    @property
    def bbox_norm(self):
        return (self.x1, self.y1, self.x2, self.y2)


# ════════════════════════════════════════════════════════
#  YOLO 偵測器（懶載入）
# ════════════════════════════════════════════════════════
class YoloDetector:
    def __init__(self, weights="yolov8n.pt", device="cpu", conf=0.35):
        self.weights = weights
        self.device  = device
        self.conf    = conf
        self._model  = None

    def _load(self):
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self.weights)
            log.info("[YOLO] 模型載入完成：%s", self.weights)
        return self._model

    def infer(self, frame: np.ndarray) -> List[Detection]:
        res = self._load().predict(
            frame, device=self.device, conf=self.conf, verbose=False)[0]
        out: List[Detection] = []
        if res.boxes is None:
            return out
        for b in res.boxes:
            cls = int(b.cls[0].item())
            x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
            out.append(Detection(res.names[cls],
                                 float(b.conf[0].item()),
                                 x1, y1, x2, y2))
        return out


# ════════════════════════════════════════════════════════
#  MediaPipe 手部偵測器（懶載入）
# ════════════════════════════════════════════════════════
class HandsDetector:
    def __init__(self, max_num_hands=1):
        self._hands = None
        self._max   = max_num_hands

    def _load(self):
        if self._hands is None:
            import mediapipe as mp
            self._hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=self._max,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            log.info("[MediaPipe] Hands 載入完成")
        return self._hands

    def detect(self, frame: np.ndarray) -> Optional[HandResult]:
        res = self._load().process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not res.multi_hand_landmarks:
            return None
        lm  = res.multi_hand_landmarks[0].landmark
        xs  = [l.x for l in lm]
        ys  = [l.y for l in lm]
        return HandResult(lm[8].x, lm[8].y,
                          min(xs), min(ys), max(xs), max(ys))

    def close(self):
        if self._hands:
            try:    self._hands.close()
            except: pass
            self._hands = None


# ════════════════════════════════════════════════════════
#  狀態機
# ════════════════════════════════════════════════════════
class FIState(Enum):
    WAITING_FOR_COMMAND = auto()
    SEARCHING_OBJECT    = auto()
    GUIDING_HEAD        = auto()
    GUIDING_HAND        = auto()
    GRAB_SUCCESS        = auto()


def _bbox_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0., ix2 - ix1)
    ih = max(0., iy2 - iy1)
    inter = iw * ih
    if inter <= 0.: return 0.
    ua    = max(0., ax2-ax1) * max(0., ay2-ay1)
    ub    = max(0., bx2-bx1) * max(0., by2-by1)
    union = ua + ub - inter
    return inter / union if union > 0 else 0.


class FindItemFSM:
    # ── 設定值（與 aiglass2 find_grab.py 一致）──
    HEAD_LEFT   = 0.40
    HEAD_RIGHT  = 0.60
    CENTER_LO   = 0.35
    CENTER_HI   = 0.65
    CENTER_REQ  = 10
    HAND_TOL    = 0.08
    GRAB_R      = 0.10
    IOU_THRESH  = 0.30
    SUCCESS_S   = 3.0
    HEAD_INT    = 1.2
    HAND_INT    = 0.6
    SEARCH_INT  = 2.0
    OBJ_MEM     = 2.0

    def __init__(self, tts_fn: Callable[[str], None]):
        import concurrent.futures
        self._tts  = tts_fn
        self._exec = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        self.state      = FIState.WAITING_FOR_COMMAND
        self.target_zh: Optional[str] = None
        self.target_en: Optional[str] = None
        self.streak     = 0
        self.last_obj:  Optional[Detection] = None
        self.last_obj_t = 0.
        self.last_stt   = ""
        self.last_stt_t = 0.
        # 計時器
        self._t_head = self._t_hand = self._t_search = self._t_success = 0.

    # ── 外部介面 ──────────────────────────────
    def set_target(self, zh: str, en: str):
        self.target_zh, self.target_en = zh, en
        self.streak = 0
        self.last_obj = None
        self._t_head = self._t_hand = self._t_search = 0.
        self._goto(FIState.SEARCHING_OBJECT)
        log.info("[FSM] 目標設定：%s (%s)", zh, en)

    def confirm(self):
        if self.state != FIState.WAITING_FOR_COMMAND:
            self._say(f"好的，已確認取得{self.target_zh or '物品'}！")
            self._reset()

    def on_speech(self, text: str):
        """外部 STT 文字 → 狀態機（背景執行緒做語意分析）"""
        self.last_stt   = text
        self.last_stt_t = time.monotonic()
        self._exec.submit(self._process_speech, text)

    def is_idle(self) -> bool:
        return self.state == FIState.WAITING_FOR_COMMAND

    def close(self):
        self._exec.shutdown(wait=False)

    # ── 每幀主入口 ────────────────────────────
    def step(self, frame: np.ndarray,
             dets: List[Detection],
             hand: Optional[HandResult]) -> None:
        if frame is None: return
        h, w = frame.shape[:2]
        now  = time.monotonic()

        if self.state == FIState.WAITING_FOR_COMMAND:
            return
        if self.state == FIState.GRAB_SUCCESS:
            if now - self._t_success >= self.SUCCESS_S:
                self._reset()
            return

        # 物品記憶
        cur = self._find_target(dets)
        if cur:
            self.last_obj = cur
            self.last_obj_t = now
            obj = cur
        elif self.last_obj and (now - self.last_obj_t) <= self.OBJ_MEM:
            obj = self.last_obj
        else:
            self.last_obj = None
            obj = None

        if   self.state == FIState.SEARCHING_OBJECT: self._tick_search(obj, now)
        elif self.state == FIState.GUIDING_HEAD:      self._tick_head(obj, w, h, now)
        elif self.state == FIState.GUIDING_HAND:      self._tick_hand(obj, hand, w, h, now)

    # ── 狀態 tick ────────────────────────────
    def _tick_search(self, obj, now):
        if obj is None:
            if now - self._t_search >= self.SEARCH_INT:
                self._t_search = now
                self._say(f"正在尋找{self.target_zh}，請慢慢轉動方向。")
            return
        self._say(f"已發現{self.target_zh}，正在引導方向。")
        self._goto(FIState.GUIDING_HEAD)

    def _tick_head(self, obj, w, h, now):
        if obj is None:
            self.streak = 0
            self._say(f"失去{self.target_zh}，重新搜尋。")
            self._goto(FIState.SEARCHING_OBJECT)
            return
        xn = obj.cx / w
        yn = obj.cy / h
        if (self.CENTER_LO <= xn <= self.CENTER_HI and
                self.CENTER_LO <= yn <= self.CENTER_HI):
            self.streak += 1
        else:
            self.streak = 0
        if now - self._t_head >= self.HEAD_INT:
            self._t_head = now
            if   xn < self.HEAD_LEFT:  self._say(f"{self.target_zh}在你的左邊，請向左轉。")
            elif xn > self.HEAD_RIGHT: self._say(f"{self.target_zh}在你的右邊，請向右轉。")
            else:                       self._say(f"{self.target_zh}就在你正前方，伸手可及。")
        if self.streak >= self.CENTER_REQ:
            self._say("目標已置中，請伸手取物。")
            self._goto(FIState.GUIDING_HAND)

    def _tick_hand(self, obj, hand, w, h, now):
        if obj is None:
            self._say(f"失去{self.target_zh}，退回引導方向。")
            self.streak = 0
            self._goto(FIState.GUIDING_HEAD)
            return
        if hand is None:
            if now - self._t_hand >= self.HAND_INT:
                self._t_hand = now
                self._say("找不到你的手，請把手伸進畫面。")
            return

        ox   = obj.cx / w;  oy = obj.cy / h
        dx   = ox - hand.tip_x
        dy   = oy - hand.tip_y
        dist = (dx*dx + dy*dy) ** 0.5
        iou  = _bbox_iou(
            (obj.x1/w, obj.y1/h, obj.x2/w, obj.y2/h),
            hand.bbox_norm
        )

        if dist < self.GRAB_R or iou >= self.IOU_THRESH:
            if now - self._t_hand >= self.HAND_INT:
                self._t_hand = now
                self._say("快到了，有沒有拿到？說「找到了」來結束。")
            return

        if now - self._t_hand < self.HAND_INT:
            return
        self._t_hand = now
        msgs = []
        if   dx >  self.HAND_TOL: msgs.append("手向右移")
        elif dx < -self.HAND_TOL: msgs.append("手向左移")
        if   dy >  self.HAND_TOL: msgs.append("手向下移")
        elif dy < -self.HAND_TOL: msgs.append("手向上移")
        self._say(("，".join(msgs) or "繼續靠近") + "。")

    # ── 工具 ─────────────────────────────────
    def _find_target(self, dets: List[Detection]) -> Optional[Detection]:
        if not self.target_en: return None
        cands = [d for d in dets if d.label.lower() == self.target_en.lower()]
        return max(cands, key=lambda d: d.conf) if cands else None

    def _goto(self, s: FIState):
        if s == self.state: return
        self.state = s
        if s == FIState.GRAB_SUCCESS:
            self._t_success = time.monotonic()

    def _reset(self):
        self.state      = FIState.WAITING_FOR_COMMAND
        self.target_zh  = self.target_en = None
        self.streak     = 0
        self.last_obj   = None
        self._t_head = self._t_hand = self._t_search = self._t_success = 0.
        log.info("[FSM] 已重置，等待指令")

    def _say(self, text: str):
        if text:
            self._exec.submit(self._tts, text)

    def _process_speech(self, text: str):
        """背景執行緒：快速關鍵字 + 選配 Groq LLM 語意分析"""
        confirm_kw = ["拿到", "確認", "完成", "找到", "到了", "好了"]
        stop_kw    = ["結束", "停止", "不要", "關閉", "停", "取消"]

        if any(k in text for k in confirm_kw):
            if self.state != FIState.WAITING_FOR_COMMAND:
                self.confirm()
            return
        if any(k in text for k in stop_kw):
            if self.state != FIState.WAITING_FOR_COMMAND:
                self._reset()
            return

        # 選配 Groq LLM（若設有 GROQ_API_KEY）
        groq_key = os.environ.get("GROQ_API_KEY", "")
        if not groq_key:
            log.warning("[FSM] GROQ_API_KEY 未設定，跳過 LLM 解析：%s", text)
            return
        try:
            import json
            from groq import Groq
            client = Groq(api_key=groq_key)
            prompt = (
                '你是智慧眼鏡指令分析大腦。若使用者想找物品，回傳 '
                '{"intent":"FIND_ITEM","target_zh":"中文名","target_en":"英文YOLO類別"}，'
                '否則回傳 {"intent":"UNKNOWN"}。只回 JSON，不要其他文字。\n'
                f'使用者說：「{text}」'
            )
            res = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            data = json.loads(res.choices[0].message.content)
            if data.get("intent") == "FIND_ITEM":
                zh = data.get("target_zh", "")
                en = data.get("target_en", "")
                if zh and en:
                    self.set_target(zh, en)
        except Exception as e:
            log.warning("[FSM] LLM 解析失敗：%s", e)


# ════════════════════════════════════════════════════════
#  字體快取（避免每幀讀硬碟，與 aiglass2 相同）
# ════════════════════════════════════════════════════════
_FONT_CACHE: dict = {}

def _put_chinese(img: np.ndarray, text: str, pos,
                 color=(0, 255, 255), size=24) -> np.ndarray:
    if not text: return img
    pil  = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    if size not in _FONT_CACHE:
        try:
            _FONT_CACHE[size] = ImageFont.truetype("msjh.ttc", size)
        except IOError:
            _FONT_CACHE[size] = ImageFont.load_default()
    draw.text(pos, text, font=_FONT_CACHE[size], fill=color)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


# ════════════════════════════════════════════════════════
#  疊加繪製（1:1 複製 aiglass2 draw_overlay）
# ════════════════════════════════════════════════════════
def draw_overlay(frame: np.ndarray, fsm: FindItemFSM,
                 dets: List[Detection],
                 target_det: Optional[Detection],
                 hand: Optional[HandResult]) -> np.ndarray:
    h, w = frame.shape[:2]

    # 物品框
    for d in dets:
        color = (0, 255, 255) if d is target_det else (100, 200, 100)
        cv2.rectangle(frame, (d.x1, d.y1), (d.x2, d.y2), color, 2)
        cv2.putText(frame, f"{d.label} {d.conf:.2f}",
                    (d.x1, max(18, d.y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # 手部框
    if hand:
        hx1 = int(hand.x1 * w); hy1 = int(hand.y1 * h)
        hx2 = int(hand.x2 * w); hy2 = int(hand.y2 * h)
        cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (255, 0, 255), 2)
        tip = (int(hand.tip_x * w), int(hand.tip_y * h))
        cv2.circle(frame, tip, 6, (255, 0, 255), -1)
        cv2.putText(frame, "hand", (hx1, max(18, hy1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA)

    # 中央參考框
    cv2.rectangle(frame,
                  (int(.35 * w), int(.35 * h)),
                  (int(.65 * w), int(.65 * h)),
                  (80, 80, 80), 1)

    # 頂端狀態列（黑底綠字）
    banner = (f"STATE: {fsm.state.name}  "
              f"target: {fsm.target_zh or '-'}  "
              f"streak: {fsm.streak}")
    cv2.rectangle(frame, (0, 0), (w, 24), (0, 0, 0), -1)
    cv2.putText(frame, banner, (8, 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    # 底部 STT 字幕（停留 4 秒）
    now = time.monotonic()
    if fsm.last_stt and (now - fsm.last_stt_t) < 4.0:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 50), (w, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        frame = _put_chinese(frame, f"語音：{fsm.last_stt}",
                             (20, h - 40), color=(0, 255, 255), size=24)
    return frame


# ════════════════════════════════════════════════════════
#  共享幀（主執行緒 ↔ 來源執行緒）
# ════════════════════════════════════════════════════════
_latest_frame:      Optional[np.ndarray] = None
_latest_frame_lock  = threading.Lock()
_stop_flag          = threading.Event()


def _set_frame(f: np.ndarray):
    global _latest_frame
    with _latest_frame_lock:
        _latest_frame = f


def _get_frame() -> Optional[np.ndarray]:
    with _latest_frame_lock:
        return None if _latest_frame is None else _latest_frame.copy()


# ════════════════════════════════════════════════════════
#  影像來源 A：本機 Webcam
# ════════════════════════════════════════════════════════
def _webcam_producer(index: int = 0):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        log.error("Webcam 開啟失敗 (index=%d)", index)
        _stop_flag.set()
        return
    log.info("[SRC] Webcam %d 已連線", index)
    try:
        while not _stop_flag.is_set():
            ok, frame = cap.read()
            if ok:  _set_frame(frame)
            else:   time.sleep(0.01)
    finally:
        cap.release()


# ════════════════════════════════════════════════════════
#  影像來源 B：WebSocket（接 aiglass3 /ws/viewer）
# ════════════════════════════════════════════════════════
def _ws_producer(url: str):
    """
    連線到 aiglass3 app_main.py 的 /ws/viewer 廣播端點，
    接收 JPEG bytes → 解碼 BGR frame。自動重連。
    """
    async def _loop():
        import websockets
        while not _stop_flag.is_set():
            try:
                log.info("[SRC] 連線至 %s ...", url)
                async with websockets.connect(
                    url,
                    max_size=8 * 1024 * 1024,
                    ping_interval=20,
                    ping_timeout=20,
                ) as ws:
                    log.info("[SRC] WebSocket 連線成功：%s", url)
                    async for msg in ws:
                        if _stop_flag.is_set(): break
                        if not isinstance(msg, (bytes, bytearray)): continue
                        arr   = np.frombuffer(msg, dtype=np.uint8)
                        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        if frame is not None:
                            _set_frame(frame)
            except Exception as e:
                log.warning("[SRC] WebSocket 中斷：%s，3 秒後重試...", e)
                await asyncio.sleep(3)

    asyncio.run(_loop())


# ════════════════════════════════════════════════════════
#  麥克風 VAD（精簡版 MicListener）
# ════════════════════════════════════════════════════════
class _MicListener:
    def __init__(self, on_text: Callable[[str], None],
                 model_size: str = "tiny",
                 sample_rate: int = 16000,
                 energy_thresh: int = 500,
                 silence_s: float = 0.8,
                 max_utt_s: float = 8.0):
        self.on_text       = on_text
        self.sample_rate   = sample_rate
        self.chunk         = int(sample_rate * 0.1)
        self.energy_thresh = energy_thresh
        self.silence_s     = silence_s
        self.max_utt_s     = max_utt_s
        self._model_size   = model_size
        self._stt          = None
        self._stop         = threading.Event()
        self._thread       = threading.Thread(
            target=self._run, daemon=True, name="mic")

    def start(self): self._thread.start()
    def stop(self):  self._stop.set()

    def _load_stt(self):
        # 優先嘗試 aiglass2 的 WhisperSTT
        try:
            sys.path.insert(0, str(Path(__file__).parent))
            from audio.stt import WhisperSTT
            return WhisperSTT(model_size=self._model_size, language="zh")
        except ImportError:
            pass
        # 退回到 openai-whisper
        try:
            import whisper as _w
            m = _w.load_model(self._model_size)
            class _Wrap:
                def transcribe_pcm(self, pcm, sr):
                    a = pcm.astype(np.float32) / 32768.
                    return _w.transcribe(m, a, language="zh")["text"]
            return _Wrap()
        except Exception as e:
            log.error("[MIC] Whisper 載入失敗：%s", e)
            return None

    def _run(self):
        try:
            import sounddevice as sd
        except ImportError:
            log.error("[MIC] sounddevice 未安裝。pip install sounddevice")
            return

        stt = self._load_stt()
        if stt is None: return

        buf: list      = []
        speaking       = False
        last_t = 0.;  utt_start = 0.
        log.info("[MIC] 開始監聽 (energy_thresh=%d)", self.energy_thresh)

        try:
            with sd.InputStream(samplerate=self.sample_rate, channels=1,
                                 dtype="int16", blocksize=self.chunk) as stream:
                while not self._stop.is_set():
                    block, _ = stream.read(self.chunk)
                    pcm    = block[:, 0] if block.ndim == 2 else block
                    energy = int(np.abs(pcm).mean())
                    now    = time.monotonic()

                    if energy > self.energy_thresh:
                        if not speaking:
                            speaking = True; utt_start = now; buf.clear()
                        last_t = now; buf.append(pcm.copy())
                    elif speaking:
                        buf.append(pcm.copy())
                        if (now - last_t > self.silence_s or
                                now - utt_start > self.max_utt_s):
                            audio   = np.concatenate(buf)
                            buf     = []; speaking = False
                            if len(audio) > self.sample_rate * 0.3:
                                try:
                                    text = (stt.transcribe_pcm(
                                        audio, self.sample_rate) or "").strip()
                                    if text:
                                        log.info("[MIC] 聽到：%s", text)
                                        self.on_text(text)
                                except Exception as e:
                                    log.warning("[MIC] STT 失敗：%s", e)
        except Exception:
            log.exception("[MIC] 執行緒崩潰")


# ════════════════════════════════════════════════════════
#  主函式
# ════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser(description="aiglass3 獨立尋物視窗 (方案A)")
    p.add_argument("--source",       choices=["webcam", "ws"], default="webcam",
                   help="影像來源：webcam 或 ws（接 aiglass3 /ws/viewer）")
    p.add_argument("--ws-url",       default="ws://127.0.0.1:8765/ws/viewer",
                   help="WebSocket URL（--source ws 時使用）")
    p.add_argument("--webcam-index", type=int, default=0)
    p.add_argument("--yolo-weights", default="yolov8n.pt")
    p.add_argument("--yolo-device",  default="cpu")
    p.add_argument("--yolo-conf",    type=float, default=0.35)
    p.add_argument("--tts",          choices=["local", "print"], default="print",
                   help="TTS 後端：local = aiglass3 play_voice_text；print = 只印 terminal")
    p.add_argument("--mic",          action="store_true",
                   help="啟用 PC 麥克風語音辨識")
    p.add_argument("--stt-model",    default="tiny")
    p.add_argument("--log-level",    default="INFO")
    args = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    # ── TTS 後端 ──────────────────────────────
    if args.tts == "local":
        try:
            from audio_player import play_voice_text, initialize_audio_system
            initialize_audio_system()
            tts_fn = play_voice_text
            log.info("[TTS] 使用 aiglass3 play_voice_text")
        except ImportError:
            log.warning("[TTS] 無法匯入 audio_player，改用 print 模式")
            tts_fn = lambda t: print(f"[TTS] {t}", flush=True)
    else:
        tts_fn = lambda t: print(f"[TTS] {t}", flush=True)

    # ── 初始化偵測器與狀態機 ──────────────────
    detector = YoloDetector(args.yolo_weights, args.yolo_device, args.yolo_conf)
    hands    = HandsDetector(max_num_hands=1)
    fsm      = FindItemFSM(tts_fn=tts_fn)

    # ── 影像來源執行緒 ────────────────────────
    if args.source == "webcam":
        t_src = threading.Thread(target=_webcam_producer,
                                 args=(args.webcam_index,),
                                 daemon=True, name="webcam_src")
    else:
        t_src = threading.Thread(target=_ws_producer,
                                 args=(args.ws_url,),
                                 daemon=True, name="ws_src")
    t_src.start()

    # ── 麥克風（選配）────────────────────────
    mic: Optional[_MicListener] = None
    if args.mic:
        mic = _MicListener(on_text=fsm.on_speech,
                           model_size=args.stt_model)
        mic.start()
        log.info("[MIC] 麥克風已啟動，請說「幫我找手機」或「找到了」")
    else:
        log.info("[TIP] 麥克風未啟用，可按視窗內 1~7 快捷鍵直接設定目標")

    # ── 建立視窗（與 aiglass2 相同）──────────
    WIN = "Find & Grab — aiglass3  (q 離開)"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 800, 600)

    log.info("[READY] 尋物視窗已就緒，等待指令")
    tts_fn("尋物模式已啟動，請按鍵選擇目標或說出指令。")

    # ── 交替幀開關（與 aiglass2 相同，降低 CPU 競爭）──
    frame_toggle = True
    last_dets: List[Detection]      = []
    last_hand: Optional[HandResult] = None

    try:
        while not _stop_flag.is_set():
            frame = _get_frame()
            if frame is None:
                time.sleep(0.01)
                cv2.waitKey(1)
                continue

            frame = cv2.flip(frame, 1)
            frame = np.ascontiguousarray(frame)

            run_hands = (fsm.state == FIState.GUIDING_HAND)

            if not run_hands:
                # 未進入手部引導：全力跑 YOLO
                dets = detector.infer(frame.copy())
                last_dets = dets
                hand      = None
                last_hand = None
            else:
                # 進入手部引導：交替幀，不讓 YOLO / MediaPipe 同幀競爭
                if frame_toggle:
                    dets      = detector.infer(frame.copy())
                    last_dets = dets
                    hand      = last_hand
                else:
                    dets      = last_dets
                    hand      = hands.detect(
                        np.array(frame, copy=True, order="C"))
                    last_hand = hand
                frame_toggle = not frame_toggle

            # 狀態機更新
            fsm.step(frame, dets, hand)

            # 繪製疊加
            target_det = fsm._find_target(dets)
            overlay    = draw_overlay(frame, fsm, dets, target_det, hand)
            cv2.imshow(WIN, overlay)

            # 鍵盤控制
            k = cv2.waitKey(1) & 0xFF
            if k in (ord('q'), 27):          # 離開
                break
            elif k == ord('r'):              # 重置
                fsm._reset()
            elif k == ord('c'):              # 確認拿到
                fsm.confirm()
            elif k == ord('t'):              # 測試：杯子
                fsm.set_target("杯子", "cup")
            elif k == ord('1'):
                fsm.set_target("水壺", "bottle")
            elif k == ord('2'):
                fsm.set_target("手機", "cell phone")
            elif k == ord('3'):
                fsm.set_target("筆電", "laptop")
            elif k == ord('4'):
                fsm.set_target("碗",   "bowl")
            elif k == ord('5'):
                fsm.set_target("書",   "book")
            elif k == ord('6'):
                fsm.set_target("鍵盤", "keyboard")
            elif k == ord('7'):
                fsm.set_target("滑鼠", "mouse")

    finally:
        _stop_flag.set()
        if mic:    mic.stop()
        hands.close()
        fsm.close()
        cv2.destroyAllWindows()
        log.info("[SHUTDOWN] 已關閉")


if __name__ == "__main__":
    main()
