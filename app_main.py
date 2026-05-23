# app_main.py
# -*- coding: utf-8 -*-
# 導入我們剛寫好的語音引擎
#python find_item_window.py --source ws --tts local --mic
from api.voice_engine import recognize_audio_from_file, text_to_speech_file
from asr_core import process_voice_file_to_ai  # 這是剛才在 asr_core 新增的函數
import os, sys, time, json, asyncio, base64, audioop
from typing import Any, Dict, Optional, Tuple, List, Callable, Set, Deque
from collections import deque
from dataclasses import dataclass
import re
import audioop
import wave
import tempfile
# 在其它 import 之后加：
from qwen_extractor import extract_english_label
from navigation_master import NavigationMaster, OrchestratorResult 
# 新增：导入盲道导航器
from workflow_blindpath import BlindPathNavigator
# 新增：导入过马路导航器
from workflow_crossstreet import CrossStreetNavigator
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketState
import uvicorn
import cv2
import numpy as np
from ultralytics import YOLO
from obstacle_detector_client import ObstacleDetectorClient

import mediapipe as mp
import bridge_io
import threading
import yolomedia  # 确保和 app_main.py 同目录，文件名就是 yolomedia.py

# ---- Windows 事件循环策略 ----
if sys.platform.startswith("win"):
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

# ---- .env ----
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ---- DashScope ASR 基础 ----
from dashscope import audio as dash_audio  # 若未安装，会在原项目里抛错提示

# groq api
from groq import AsyncGroq
groq_client = AsyncGroq(api_key="")

MODEL        = "paraformer-realtime-v2"
SAMPLE_RATE  = 16000
AUDIO_FMT    = "pcm"
CHUNK_MS     = 20
BYTES_CHUNK  = SAMPLE_RATE * CHUNK_MS // 1000 * 2
SILENCE_20MS = bytes(BYTES_CHUNK)

# ---- 引入我们的模块 ----
from audio_stream import (
    register_stream_route,
    broadcast_pcm16_realtime,
    hard_reset_audio,
    BYTES_PER_20MS_16K,
    is_playing_now,
    current_ai_task,
)
from omni_client import stream_chat, OmniStreamPiece
from asr_core import (
    ASRCallback,
    set_current_recognition,
    stop_current_recognition,
)
from audio_player import initialize_audio_system, play_voice_text

# ---- 同步录制器 ----
import sync_recorder
import signal
import atexit

# ---- IMU UDP ----
UDP_IP   = "0.0.0.0"
UDP_PORT = 12345

app = FastAPI()

# ====== 状态与容器 ======
app.mount("/static", StaticFiles(directory="static"), name="static")

ui_clients: Dict[int, WebSocket] = {}
current_partial: str = ""
recent_finals: List[str] = []
RECENT_MAX = 50
last_frames: Deque[Tuple[float, bytes]] = deque(maxlen=10)

camera_viewers: Set[WebSocket] = set()
find_window_control_clients: Set[WebSocket] = set()
esp32_camera_ws: Optional[WebSocket] = None
imu_ws_clients: Set[WebSocket] = set()
esp32_audio_ws: Optional[WebSocket] = None

blind_path_navigator = None
navigation_active = False
yolo_seg_model = None
obstacle_detector = None

cross_street_navigator = None
cross_street_active = False
orchestrator = None

omni_conversation_active = False
omni_previous_nav_state = None

def load_navigation_models():
    global yolo_seg_model, obstacle_detector
    try:
        seg_model_path = os.getenv("BLIND_PATH_MODEL", r"model\yolo-seg.pt")
        if os.path.exists(seg_model_path):
            print(f"[NAVIGATION] 模型文件存在，开始加载...")
            yolo_seg_model = YOLO(seg_model_path)
            if torch.cuda.is_available():
                yolo_seg_model.to("cuda")
                print(f"[NAVIGATION] 盲道分割模型加载成功并放到GPU: {yolo_seg_model.device}")
            else:
                print("[NAVIGATION] CUDA不可用，模型仍在CPU")
            try:
                test_img = np.zeros((640, 640, 3), dtype=np.uint8)
                yolo_seg_model.predict(test_img, device="cuda" if torch.cuda.is_available() else "cpu", verbose=False)
                print(f"[NAVIGATION] 模型测试成功")
            except Exception as e:
                print(f"[NAVIGATION] 模型测试失败: {e}")
        else:
            print(f"[NAVIGATION] 错误：找不到模型文件: {seg_model_path}")

        obstacle_model_path = os.getenv("OBSTACLE_MODEL", r"model\yoloe-11l-seg.pt")
        if os.path.exists(obstacle_model_path):
            try:
                obstacle_detector = ObstacleDetectorClient(model_path=obstacle_model_path)
                print(f"[NAVIGATION] ========== YOLO-E 障碍物检测器加载成功 ==========")
            except Exception as e:
                print(f"[NAVIGATION] 障碍物检测器加载失败: {e}")
                import traceback; traceback.print_exc()
                obstacle_detector = None
        else:
            print(f"[NAVIGATION] 警告：找不到障碍物检测模型文件: {obstacle_model_path}")
    except Exception as e:
        print(f"[NAVIGATION] 模型加载失败: {e}")
        import traceback; traceback.print_exc()

print("[NAVIGATION] 开始加载导航模型...")
load_navigation_models()
print(f"[NAVIGATION] 模型加载完成 - yolo_seg_model: {yolo_seg_model is not None}")

print("[RECORDER] 启动同步录制系统...")
sync_recorder.start_recording()
print("[RECORDER] 录制系统已启动，将自动保存视频和音频")

def cleanup_on_exit():
    print("\n[SYSTEM] 正在关闭录制器...")
    try:
        sync_recorder.stop_recording()
        print("[SYSTEM] 录制文件已保存")
    except Exception as e:
        print(f"[SYSTEM] 关闭录制器时出错: {e}")

def signal_handler(sig, frame):
    print("\n[SYSTEM] 收到中断信号，正在安全退出...")
    cleanup_on_exit()
    sys.exit(0)

# ── 關鍵修正：signal.signal() 只能在主執行緒呼叫 ──
# 當被 gui_window.py 在子執行緒 import 時跳過，避免 ValueError
import threading as _threading
if _threading.current_thread() is _threading.main_thread():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
atexit.register(cleanup_on_exit)

print("[RECORDER] 已注册退出处理器 - Ctrl+C时会自动保存录制文件")

try:
    import trafficlight_detection
    print("[TRAFFIC_LIGHT] 开始预加载红绿灯检测模型...")
    if trafficlight_detection.init_model():
        print("[TRAFFIC_LIGHT] 红绿灯检测模型预加载成功")
        try:
            test_img = np.zeros((640, 640, 3), dtype=np.uint8)
            _ = trafficlight_detection.process_single_frame(test_img)
            print("[TRAFFIC_LIGHT] 模型预热完成")
        except Exception as e:
            print(f"[TRAFFIC_LIGHT] 模型预热失败: {e}")
    else:
        print("[TRAFFIC_LIGHT] 红绿灯检测模型预加载失败")
except Exception as e:
    print(f"[TRAFFIC_LIGHT] 红绿灯模型预加载出错: {e}")

interrupt_lock = asyncio.Lock()

yolomedia_thread: Optional[threading.Thread] = None
yolomedia_stop_event = threading.Event()
yolomedia_running = False
yolomedia_sending_frames = False

ITEM_TO_CLASS_MAP = {
    "红牛": "Red_Bull",
    "AD钙奶": "AD_milk",
    "ad钙奶": "AD_milk",
    "钙奶": "AD_milk",
}

async def ui_broadcast_raw(msg: str):
    dead = []
    for k, ws in list(ui_clients.items()):
        try:
            await ws.send_text(msg)
        except Exception:
            dead.append(k)
    for k in dead:
        ui_clients.pop(k, None)

async def ui_broadcast_partial(text: str):
    global current_partial
    current_partial = text
    await ui_broadcast_raw("PARTIAL:" + text)

async def ui_broadcast_final(text: str):
    global current_partial, recent_finals
    current_partial = ""
    recent_finals.append(text)
    if len(recent_finals) > RECENT_MAX:
        recent_finals = recent_finals[-RECENT_MAX:]
    await ui_broadcast_raw("FINAL:" + text)
    print(f"[ASR/AI FINAL] {text}", flush=True)

async def full_system_reset(reason: str = ""):
    global current_partial, recent_finals, orchestrator, yolomedia_running
    await hard_reset_audio(reason or "full_system_reset")
    await stop_current_recognition()
    if orchestrator:
        orchestrator.stop_navigation()
        print(f"[SYSTEM] 導航已從系統重置中強制關閉 (原因: {reason})")
    if yolomedia_running:
        stop_yolomedia()
    global current_partial, recent_finals
    current_partial = ""
    recent_finals = []
    try:
        last_frames.clear()
    except Exception:
        pass
    try:
        if esp32_audio_ws and (esp32_audio_ws.client_state == WebSocketState.CONNECTED):
            await esp32_audio_ws.send_text("RESET")
    except Exception:
        pass
    print("[SYSTEM] full reset done.", flush=True)
    
_find_window_fsm = None  # find_item_window.py 啟動後會注入

def register_find_window_fsm(fsm):
    """讓 find_item_window.py 在啟動時把 FSM 物件注入到 app_main"""
    global _find_window_fsm
    _find_window_fsm = fsm
    print("[APP_MAIN] find_item_window FSM 已注冊", flush=True)

def _notify_find_window(zh: str, en: str):
    global _find_window_fsm
    if _find_window_fsm is not None:
        try:
            _find_window_fsm.set_target(zh, en)
            print(f"[APP_MAIN] 已通知 find_window FSM：{zh} ({en})", flush=True)
        except Exception as e:
            print(f"[APP_MAIN] 通知 find_window FSM 失敗：{e}", flush=True)
            
async def _broadcast_find_window_command(payload: Dict[str, Any]):
    if not find_window_control_clients:
        return
    msg = json.dumps(payload, ensure_ascii=False)
    dead = []
    for ws in list(find_window_control_clients):
        try:
            await ws.send_text(msg)
        except Exception:
            dead.append(ws)
    for ws in dead:
        find_window_control_clients.discard(ws)

def _schedule_find_window_command(payload: Dict[str, Any]):
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_broadcast_find_window_command(payload))
    except RuntimeError:
        pass

def _notify_find_window_mode(mode: str):
    global _find_window_fsm
    if _find_window_fsm is not None:
        try:
            if hasattr(_find_window_fsm, "set_app_mode"):
                _find_window_fsm.set_app_mode(mode)
            elif mode != "FIND_ITEM" and hasattr(_find_window_fsm, "_reset"):
                _find_window_fsm._reset()
        except Exception as e:
            print(f"[APP_MAIN] find_window mode notify failed: {e}", flush=True)
    _schedule_find_window_command({"type": "mode", "mode": mode})

def _notify_find_window(zh: str, en: str):
    global _find_window_fsm
    if _find_window_fsm is not None:
        try:
            if hasattr(_find_window_fsm, "set_app_mode"):
                _find_window_fsm.set_app_mode("FIND_ITEM")
            _find_window_fsm.set_target(zh, en)
        except Exception as e:
            print(f"[APP_MAIN] find_window target notify failed: {e}", flush=True)
    _schedule_find_window_command({"type": "find_target", "zh": zh, "en": en})

def start_yolomedia_with_target(target_name: str):
    global yolomedia_thread, yolomedia_stop_event, yolomedia_running, yolomedia_sending_frames
    if yolomedia_running:
        stop_yolomedia()
    yolo_class = ITEM_TO_CLASS_MAP.get(target_name, target_name)
    print(f"[YOLOMEDIA] Starting with target: {target_name} -> YOLO class: {yolo_class}", flush=True)
    yolomedia_stop_event.clear()
    yolomedia_running = True
    yolomedia_sending_frames = False
    def _run():
        try:
            yolomedia.main(headless=True, prompt_name=yolo_class, stop_event=yolomedia_stop_event)
        except Exception as e:
            print(f"[YOLOMEDIA] worker stopped: {e}", flush=True)
        finally:
            global yolomedia_running, yolomedia_sending_frames
            yolomedia_running = False
            yolomedia_sending_frames = False
    yolomedia_thread = threading.Thread(target=_run, daemon=True)
    yolomedia_thread.start()
    print(f"[YOLOMEDIA] background worker started for: {yolo_class}", flush=True)

def stop_yolomedia():
    global yolomedia_thread, yolomedia_stop_event, yolomedia_running, yolomedia_sending_frames
    if yolomedia_running:
        print("[YOLOMEDIA] Stopping worker...", flush=True)
        yolomedia_stop_event.set()
        if yolomedia_thread and yolomedia_thread.is_alive():
            yolomedia_thread.join(timeout=5.0)
        yolomedia_running = False
        yolomedia_sending_frames = False
        print("[YOLOMEDIA] Worker stopped.", flush=True)

async def analyze_intent_with_groq(user_text: str) -> str:
    stop_keywords = ["結束", "停止", "不要", "關閉", "停"]
    if any(k in user_text for k in stop_keywords):
        return "STOP"
    prompt = f"""
    你是一個盲人導航眼鏡的指令分析大腦。
    請判斷使用者的語音指令屬於哪一個類別。
    只允許回傳以下英文代碼，絕對不要回傳其他任何廢話或標點符號：
    - CROSS_STREET
    - TRAFFIC_LIGHT
    - BLIND_PATH
    - FIND_ITEM
    - STOP
    - UNKNOWN
    使用者說：「{user_text}」
    """
    try:
        response = await groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        intent = response.choices[0].message.content.strip()
        for valid_intent in ["CROSS_STREET", "TRAFFIC_LIGHT", "BLIND_PATH", "FIND_ITEM", "STOP"]:
            if valid_intent in intent:
                return valid_intent
        return "UNKNOWN"
    except Exception as e:
        print(f"[LLM Router Error] Groq API 發生錯誤: {e}")
        return "UNKNOWN"

async def start_ai_with_text_custom(user_text: str):
    global navigation_active, blind_path_navigator, cross_street_active, cross_street_navigator, orchestrator, yolomedia_running
    print(f"[COMMAND_CHECK] 收到語音指令: {user_text}")
    intent = await analyze_intent_with_groq(user_text)
    print(f"[LLM ROUTER] 判斷結果: {intent}")
    if "STOP" in intent:
        _notify_find_window_mode("STOP")
        if orchestrator:
            orchestrator.stop_navigation()
            play_voice_text("導航已停止。")
            await ui_broadcast_final("[系统] 導航已停止")
        return
    if "CROSS_STREET" in intent:
        _notify_find_window_mode("CROSS_STREET")
        if yolomedia_running: stop_yolomedia()
        if orchestrator:
            orchestrator.start_crossing()
            play_voice_text("過馬路模式已啟動。")
            await ui_broadcast_final("[系统] 過馬路模式已啟動")
        return
    if "TRAFFIC_LIGHT" in intent:
        _notify_find_window_mode("TRAFFIC_LIGHT")
        try:
            import trafficlight_detection
            if orchestrator: orchestrator.start_traffic_light_detection()
            success = trafficlight_detection.init_model()
            trafficlight_detection.reset_detection_state()
            if success:
                await ui_broadcast_final("[系统] 紅綠燈檢測已啟動")
                play_voice_text("紅綠燈檢測已啟動。")
        except Exception as e:
            print(f"[TRAFFIC] 啟動失敗: {e}")
        return
    if "BLIND_PATH" in intent:
        _notify_find_window_mode("BLIND_PATH")
        if yolomedia_running: stop_yolomedia()
        if orchestrator:
            orchestrator.start_blind_path_navigation()
            await ui_broadcast_final("[系统] 盲道導航已啟動")
            play_voice_text("盲道導航已啟動。")
        return
    if "FIND_ITEM" in intent:
        # 更寬鬆的 pattern，涵蓋「我要找」「幫我找」「找一下」「找」
        find_pattern = r"找(?:一下|一個|個|下)?\s*(.{1,10}?)(?:。|！|？|，|$)"
        match = re.search(find_pattern, user_text)
        if match:
            item_cn = match.group(1).strip()
        else:
            # fallback：把整句丟給 extractor 自己判斷
            item_cn = user_text.strip()

        label_en, src = extract_english_label(item_cn)
        print(f"[FIND_ITEM] 語音='{user_text}' → item_cn='{item_cn}' → en='{label_en}' (src={src})")

        _notify_find_window(item_cn, label_en)

        if orchestrator:
            orchestrator.start_item_search()

        start_yolomedia_with_target(label_en)
        await ui_broadcast_final(f"[找物品] 正在尋找 {item_cn}...")
        play_voice_text(f"正在尋找 {item_cn}。")
        return
    print(f"[OMNI] 未知指令，已攔截: {user_text}")
    await ui_broadcast_final(f"[系统] 未知指令: {user_text}")

async def start_ai_with_text(user_text: str):
    async def _runner():
        txt_buf: List[str] = []
        rate_state = None
        content_list = []
        if last_frames:
            try:
                _, jpeg_bytes = last_frames[-1]
                img_b64 = base64.b64encode(jpeg_bytes).decode("ascii")
                content_list.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}})
            except Exception:
                pass
        content_list.append({"type": "text", "text": user_text})
        try:
            async for piece in stream_chat(content_list, voice="Cherry", audio_format="wav"):
                if piece.text_delta:
                    txt_buf.append(piece.text_delta)
                    try: await ui_broadcast_partial("[AI] " + "".join(txt_buf))
                    except Exception: pass
                if piece.audio_b64:
                    try: pcm24 = base64.b64decode(piece.audio_b64)
                    except Exception: pcm24 = b""
                    if pcm24:
                        pcm8k, rate_state = audioop.ratecv(pcm24, 2, 1, 24000, 8000, rate_state)
                        pcm8k = audioop.mul(pcm8k, 2, 0.60)
                        if pcm8k: await broadcast_pcm16_realtime(pcm8k)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            try: await ui_broadcast_final(f"[AI] 发生错误：{e}")
            except Exception: pass
        finally:
            global omni_conversation_active, omni_previous_nav_state
            omni_conversation_active = False
            if orchestrator and omni_previous_nav_state:
                orchestrator.force_state(omni_previous_nav_state)
                omni_previous_nav_state = None
            from audio_stream import stream_clients
            for sc in list(stream_clients):
                if not sc.abort_event.is_set():
                    try: sc.q.put_nowait(b"\x00"*BYTES_PER_20MS_16K)
                    except Exception: pass
                    try: sc.q.put_nowait(None)
                    except Exception: pass
            final_text = ("".join(txt_buf)).strip() or "（空响应）"
            try: await ui_broadcast_final("[AI] " + final_text)
            except Exception: pass
    await hard_reset_audio("start_ai_with_text")
    loop = asyncio.get_running_loop()
    from audio_stream import __dict__ as _as_dict
    task = loop.create_task(_runner())
    _as_dict["current_ai_task"] = task

@app.get("/", response_class=HTMLResponse)
def root():
    with open(os.path.join("templates", "index.html"), "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

@app.get("/api/health", response_class=PlainTextResponse)
def health():
    return "OK"

register_stream_route(app)

@app.websocket("/ws_ui")
async def ws_ui(ws: WebSocket):
    await ws.accept()
    ui_clients[id(ws)] = ws
    try:
        init = {"partial": current_partial, "finals": recent_finals[-10:]}
        await ws.send_text("INIT:" + json.dumps(init, ensure_ascii=False))
        while True:
            await asyncio.sleep(60)
    except WebSocketDisconnect:
        pass
    finally:
        ui_clients.pop(id(ws), None)

@app.websocket("/ws_audio")
async def ws_audio(ws: WebSocket):
    global esp32_audio_ws
    esp32_audio_ws = ws
    await ws.accept()
    print("\n[AUDIO] client connected")
    streaming = False
    VAD_THRESHOLD = 300
    SILENCE_LIMIT_CHUNKS = 50
    is_speaking = False
    silence_counter = 0
    audio_buffer = bytearray()
    loop = asyncio.get_running_loop()
    def post(coro):
        asyncio.run_coroutine_threadsafe(coro, loop)
    cb = ASRCallback(
        on_sdk_error=lambda s: print(f"[ASR] Error: {s}"),
        post=post,
        ui_broadcast_partial=ui_broadcast_partial,
        ui_broadcast_final=ui_broadcast_final,
        is_playing_now_fn=is_playing_now,
        start_ai_with_text_fn=start_ai_with_text_custom,
        full_system_reset_fn=full_system_reset,
        interrupt_lock=interrupt_lock,
    )
    try:
        while True:
            if WebSocketState and ws.client_state != WebSocketState.CONNECTED:
                break
            try:
                msg = await ws.receive()
            except WebSocketDisconnect:
                break
            except RuntimeError as e:
                if 'Cannot call "receive"' in str(e): break
                raise
            if "text" in msg and msg["text"] is not None:
                raw = (msg["text"] or "").strip()
                cmd = raw.upper()
                if cmd == "START":
                    print("[AUDIO] START received")
                    sync_recorder.start_recording()
                    streaming = True
                    is_speaking = False
                    silence_counter = 0
                    audio_buffer.clear()
                    await ui_broadcast_partial("（已開始接收音訊…）")
                    await ws.send_text("OK:STARTED")
                elif cmd == "STOP":
                    streaming = False
                    await ws.send_text("OK:STOPPED")
                elif raw.startswith("PROMPT:"):
                    text = raw[len("PROMPT:"):].strip()
                    if text:
                        async with interrupt_lock:
                            await start_ai_with_text_custom(text)
                        await ws.send_text("OK:PROMPT_ACCEPTED")
                    else:
                        await ws.send_text("ERR:EMPTY_PROMPT")
            elif "bytes" in msg and msg["bytes"] is not None:
                pcm_data = msg["bytes"]
                sync_recorder.record_audio(pcm_data)
                if streaming:
                    rms = audioop.rms(pcm_data, 2)
                    if rms > VAD_THRESHOLD:
                        if not is_speaking:
                            print("[VAD] 偵測到語音開始...")
                        is_speaking = True
                        silence_counter = 0
                        audio_buffer.extend(pcm_data)
                    elif is_speaking:
                        silence_counter += 1
                        audio_buffer.extend(pcm_data)
                        if silence_counter > SILENCE_LIMIT_CHUNKS:
                            print("[VAD] 語音結束，準備送交 Groq Whisper 辨識")
                            is_speaking = False
                            temp_wav = tempfile.mktemp(suffix=".wav")
                            with wave.open(temp_wav, 'wb') as wf:
                                wf.setnchannels(1)
                                wf.setsampwidth(2)
                                wf.setframerate(16000)
                                wf.writeframes(audio_buffer)
                            audio_buffer.clear()
                            asyncio.create_task(process_voice_file_to_ai(temp_wav, cb))
    except Exception as e:
        print(f"\n[WS ERROR] {e}")
    finally:
        try:
            if WebSocketState is None or ws.client_state == WebSocketState.CONNECTED:
                await ws.close(code=1000)
        except Exception:
            pass
        if esp32_audio_ws is ws:
            esp32_audio_ws = None
        print("[WS] connection closed")

@app.websocket("/ws/camera")
async def ws_camera_esp(ws: WebSocket):
    global esp32_camera_ws, blind_path_navigator, cross_street_navigator, cross_street_active, navigation_active, orchestrator
    if esp32_camera_ws is not None:
        await ws.close(code=1013)
        return
    esp32_camera_ws = ws
    await ws.accept()
    print("[CAMERA] ESP32 connected")
    if blind_path_navigator is None and yolo_seg_model is not None:
        blind_path_navigator = BlindPathNavigator(yolo_seg_model, obstacle_detector)
        print("[NAVIGATION] 盲道导航器已初始化")
    if cross_street_navigator is None and yolo_seg_model:
        cross_street_navigator = CrossStreetNavigator(seg_model=yolo_seg_model, coco_model=None, obs_model=None)
        print("[CROSS_STREET] 过马路导航器已初始化")
    if orchestrator is None and blind_path_navigator is not None and cross_street_navigator is not None:
        orchestrator = NavigationMaster(blind_path_navigator, cross_street_navigator)
        print("[NAV MASTER] 统领状态机已初始化")
    frame_counter = 0
    try:
        while True:
            msg = await ws.receive()
            if "bytes" in msg and msg["bytes"] is not None:
                data = msg["bytes"]
                frame_counter += 1
                try: sync_recorder.record_frame(data)
                except Exception: pass
                try: last_frames.append((time.time(), data))
                except Exception: pass
                bridge_io.push_raw_jpeg(data)
                if frame_counter % 30 == 0:
                    state_dbg = orchestrator.get_state() if orchestrator else "N/A"
                    print(f"[NAVIGATION DEBUG] 帧:{frame_counter}, state={state_dbg}, yolomedia_running={yolomedia_running}")
                try:
                    arr = np.frombuffer(data, dtype=np.uint8)
                    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if bgr is None or bgr.size == 0: bgr = None
                except Exception:
                    bgr = None
                if orchestrator and not yolomedia_running and bgr is not None:
                    current_state = orchestrator.get_state()
                    if current_state == "ITEM_SEARCH":
                        if not yolomedia_sending_frames and camera_viewers:
                            ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                            if ok:
                                jpeg_data = enc.tobytes()
                                dead = []
                                for viewer_ws in list(camera_viewers):
                                    try: await viewer_ws.send_bytes(jpeg_data)
                                    except Exception: dead.append(viewer_ws)
                                for d in dead: camera_viewers.discard(d)
                        continue
                    out_img = bgr
                    try:
                        if current_state == "TRAFFIC_LIGHT_DETECTION":
                            import trafficlight_detection
                            result = trafficlight_detection.process_single_frame(bgr, ui_broadcast_callback=ui_broadcast_final)
                            out_img = result['vis_image'] if result['vis_image'] is not None else bgr
                        else:
                            res = orchestrator.process_frame(bgr)
                            if hasattr(res, 'display_text') and res.display_text:
                                try: asyncio.create_task(ui_broadcast_final(f"[{res.state}] {res.display_text}"))
                                except Exception: pass
                            if hasattr(res, 'guidance_text') and res.guidance_text:
                                try: play_voice_text(res.guidance_text)
                                except Exception: pass
                            out_img = res.annotated_image if res.annotated_image is not None else bgr
                    except Exception as e:
                        if frame_counter % 100 == 0: print(f"[NAV MASTER] 处理帧时出错: {e}")
                    if camera_viewers and out_img is not None:
                        ok, enc = cv2.imencode(".jpg", out_img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                        if ok:
                            jpeg_data = enc.tobytes()
                            dead = []
                            for viewer_ws in list(camera_viewers):
                                try: await viewer_ws.send_bytes(jpeg_data)
                                except Exception: dead.append(viewer_ws)
                            for d in dead: camera_viewers.discard(d)
                    continue
                if not yolomedia_sending_frames and camera_viewers and bgr is not None:
                    try:
                        ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                        if ok:
                            jpeg_data = enc.tobytes()
                            dead = []
                            for viewer_ws in list(camera_viewers):
                                try: await viewer_ws.send_bytes(jpeg_data)
                                except Exception: dead.append(viewer_ws)
                            for d in dead: camera_viewers.discard(d)
                    except Exception as e:
                        print(f"[CAMERA] Broadcast error: {e}")
            elif "type" in msg and msg["type"] in ("websocket.close", "websocket.disconnect"):
                break
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[CAMERA ERROR] {e}")
    finally:
        try:
            if WebSocketState is None or ws.client_state == WebSocketState.CONNECTED:
                await ws.close(code=1000)
        except Exception:
            pass
        esp32_camera_ws = None
        print("[CAMERA] ESP32 disconnected")
        if blind_path_navigator: blind_path_navigator.reset()
        if cross_street_navigator: cross_street_navigator.reset()
        if orchestrator:
            orchestrator.reset()
            print("[NAV MASTER] 统领器已重置")

@app.websocket("/ws/viewer")
async def ws_viewer(ws: WebSocket):
    await ws.accept()
    camera_viewers.add(ws)
    print(f"[VIEWER] Browser connected. Total viewers: {len(camera_viewers)}", flush=True)
    try:
        while True:
            await asyncio.sleep(60)
    except WebSocketDisconnect:
        print("[VIEWER] Browser disconnected", flush=True)
    finally:
        try: camera_viewers.remove(ws)
        except Exception: pass
        print(f"[VIEWER] Removed. Total viewers: {len(camera_viewers)}", flush=True)

@app.websocket("/ws/find_item_control")
async def ws_find_item_control(ws: WebSocket):
    await ws.accept()
    find_window_control_clients.add(ws)
    print(f"[FIND_WINDOW] Control connected. Total: {len(find_window_control_clients)}", flush=True)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        find_window_control_clients.discard(ws)
        print(f"[FIND_WINDOW] Control removed. Total: {len(find_window_control_clients)}", flush=True)

@app.websocket("/ws")
async def ws_imu(ws: WebSocket):
    await ws.accept()
    imu_ws_clients.add(ws)
    try:
        while True:
            await asyncio.sleep(60)
    except WebSocketDisconnect:
        pass
    finally:
        imu_ws_clients.discard(ws)

async def imu_broadcast(msg: str):
    if not imu_ws_clients: return
    dead = []
    for ws in list(imu_ws_clients):
        try: await ws.send_text(msg)
        except Exception: dead.append(ws)
    for ws in dead: imu_ws_clients.discard(ws)

from math import atan2, hypot, pi
GRAV_BETA   = 0.98
STILL_W     = 0.4
YAW_DB      = 0.08
YAW_LEAK    = 0.2
ANG_EMA     = 0.15
AUTO_REZERO = True
USE_PROJ    = True
FREEZE_STILL= True
G     = 9.807
A_TOL = 0.08 * G
gLP = {"x":0.0, "y":0.0, "z":0.0}
gOff= {"x":0.0, "y":0.0, "z":0.0}
BIAS_ALPHA = 0.002
yaw  = 0.0
Rf = Pf = Yf = 0.0
ref = {"roll":0.0, "pitch":0.0, "yaw":0.0}
holdStart = 0.0
isStill   = False
last_ts_imu = 0.0
last_wall = 0.0
imu_store: List[Dict[str, Any]] = []

def _wrap180(a: float) -> float:
    a = a % 360.0
    if a >= 180.0: a -= 360.0
    if a < -180.0: a += 360.0
    return a

def process_imu_and_maybe_store(d: Dict[str, Any]):
    global gLP, gOff, yaw, Rf, Pf, Yf, ref, holdStart, isStill, last_ts_imu, last_wall
    t_ms = float(d.get("ts", 0.0))
    now_wall = time.monotonic()
    if t_ms <= 0.0: t_ms = (now_wall * 1000.0)
    if last_ts_imu <= 0.0 or t_ms <= last_ts_imu or (t_ms - last_ts_imu) > 3000.0: dt = 0.02
    else: dt = (t_ms - last_ts_imu) / 1000.0
    last_ts_imu = t_ms
    ax = float(((d.get("accel") or {}).get("x", 0.0)))
    ay = float(((d.get("accel") or {}).get("y", 0.0)))
    az = float(((d.get("accel") or {}).get("z", 0.0)))
    wx = float(((d.get("gyro")  or {}).get("x", 0.0)))
    wy = float(((d.get("gyro")  or {}).get("y", 0.0)))
    wz = float(((d.get("gyro")  or {}).get("z", 0.0)))
    gLP["x"] = GRAV_BETA * gLP["x"] + (1.0 - GRAV_BETA) * ax
    gLP["y"] = GRAV_BETA * gLP["y"] + (1.0 - GRAV_BETA) * ay
    gLP["z"] = GRAV_BETA * gLP["z"] + (1.0 - GRAV_BETA) * az
    gmag = hypot(gLP["x"], gLP["y"], gLP["z"]) or 1.0
    gHat = {"x": gLP["x"]/gmag, "y": gLP["y"]/gmag, "z": gLP["z"]/gmag}
    roll  = (atan2(az, ay)  * 180.0 / pi)
    pitch = (atan2(-ax, ay) * 180.0 / pi)
    aNorm = hypot(ax, ay, az); wNorm = hypot(wx, wy, wz)
    nearFlat = (abs(roll) < 2.0 and abs(pitch) < 2.0)
    stillCond = (abs(aNorm - G) < A_TOL) and (wNorm < STILL_W)
    if stillCond:
        if holdStart <= 0.0: holdStart = t_ms
        if not isStill and (t_ms - holdStart) > 350.0: isStill = True
        gOff["x"] = (1.0 - BIAS_ALPHA)*gOff["x"] + BIAS_ALPHA*wx
        gOff["y"] = (1.0 - BIAS_ALPHA)*gOff["y"] + BIAS_ALPHA*wy
        gOff["z"] = (1.0 - BIAS_ALPHA)*gOff["z"] + BIAS_ALPHA*wz
    else:
        holdStart = 0.0; isStill = False
    if USE_PROJ:
        yawdot = ((wx - gOff["x"])*gHat["x"] + (wy - gOff["y"])*gHat["y"] + (wz - gOff["z"])*gHat["z"])
    else:
        yawdot = (wy - gOff["y"])
    if abs(yawdot) < YAW_DB: yawdot = 0.0
    if FREEZE_STILL and stillCond: yawdot = 0.0
    yaw = _wrap180(yaw + yawdot * dt)
    if (YAW_LEAK > 0.0) and nearFlat and stillCond and abs(yaw) > 0.0:
        step = YAW_LEAK * dt * (-1.0 if yaw > 0 else (1.0 if yaw < 0 else 0.0))
        if abs(yaw) <= abs(step): yaw = 0.0
        else: yaw += step
    global Rf, Pf, Yf, ref, last_wall
    Rf = ANG_EMA * roll  + (1.0 - ANG_EMA) * Rf
    Pf = ANG_EMA * pitch + (1.0 - ANG_EMA) * Pf
    Yf = ANG_EMA * yaw   + (1.0 - ANG_EMA) * Yf
    if AUTO_REZERO and nearFlat and (wNorm < STILL_W):
        if holdStart <= 0.0: holdStart = t_ms
        if not isStill and (t_ms - holdStart) > 350.0:
            ref.update({"roll": Rf, "pitch": Pf, "yaw": Yf})
            isStill = True
    R = _wrap180(Rf - ref["roll"])
    P = _wrap180(Pf - ref["pitch"])
    Y = _wrap180(Yf - ref["yaw"])
    now_wall = time.monotonic()
    if last_wall <= 0.0 or (now_wall - last_wall) >= 0.100:
        last_wall = now_wall
        item = {"ts": t_ms/1000.0, "angles": {"roll": R, "pitch": P, "yaw": Y}, "accel": {"x": ax, "y": ay, "z": az}, "gyro": {"x": wx, "y": wy, "z": wz}}
        imu_store.append(item)

class UDPProto(asyncio.DatagramProtocol):
    def connection_made(self, transport):
        print(f"[UDP] listening on {UDP_IP}:{UDP_PORT}")
    def datagram_received(self, data, addr):
        try:
            s = data.decode('utf-8', errors='ignore').strip()
            d = json.loads(s)
            if 'ts' not in d and 'timestamp_ms' in d: d['ts'] = d.pop('timestamp_ms')
            process_imu_and_maybe_store(d)
            asyncio.create_task(imu_broadcast(json.dumps(d)))
        except Exception:
            pass

@app.on_event("startup")
async def on_startup_register_bridge_sender():
    main_loop = asyncio.get_event_loop()
    def _sender(jpeg_bytes: bytes):
        try:
            if main_loop.is_closed(): return
            global yolomedia_sending_frames
            if not yolomedia_sending_frames:
                yolomedia_sending_frames = True
                print("[YOLOMEDIA] 开始发送处理后的帧，切换到YOLO画面", flush=True)
            async def _broadcast():
                if not camera_viewers: return
                dead = []
                for ws in list(camera_viewers):
                    try: await ws.send_bytes(jpeg_bytes)
                    except Exception: dead.append(ws)
                for ws in dead:
                    try: camera_viewers.remove(ws)
                    except Exception: pass
            asyncio.run_coroutine_threadsafe(_broadcast(), main_loop)
        except Exception as e:
            if "Event loop is closed" not in str(e):
                print(f"[DEBUG] _sender error: {e}", flush=True)
    bridge_io.set_sender(_sender)

@app.on_event("startup")
async def on_startup_init_audio():
    def _init():
        try: initialize_audio_system()
        except Exception as e: print(f"[AUDIO] 初始化失败: {e}")
    threading.Thread(target=_init, daemon=True).start()

@app.on_event("startup")
async def on_startup():
    loop = asyncio.get_running_loop()
    await loop.create_datagram_endpoint(lambda: UDPProto(), local_addr=(UDP_IP, UDP_PORT))

@app.on_event("shutdown")
async def on_shutdown():
    print("[SHUTDOWN] 开始清理资源...")
    stop_yolomedia()
    await hard_reset_audio("shutdown")
    print("[SHUTDOWN] 资源清理完成")

def get_last_frames():
    return last_frames

def get_camera_ws():
    return esp32_camera_ws

if __name__ == "__main__":
    uvicorn.run(
        app, host="0.0.0.0", port=8765,
        log_level="warning", access_log=False,
        loop="asyncio", workers=1, reload=False
    )
