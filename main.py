import os
import json
import asyncio
import logging
import numpy as np
from fastapi import FastAPI, WebSocket
from faster_whisper import WhisperModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VoxServer")

MODEL_NAME = os.getenv("WHISPER_MODEL", "small")
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "int8")
DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
CPU_THREADS = int(os.getenv("CPU_THREADS", "2"))
API_KEY = os.getenv("API_KEY", "")

FINAL_BEAM = int(os.getenv("FINAL_BEAM", "5"))
FINAL_BEST_OF = int(os.getenv("FINAL_BEST_OF", "1"))
FINAL_PATIENCE = float(os.getenv("FINAL_PATIENCE", "2.5"))
FINAL_TEMPERATURE = os.getenv("FINAL_TEMPERATURE", "0.0")
FINAL_REPETITION_PENALTY = float(os.getenv("FINAL_REPETITION_PENALTY", "1.2"))
FINAL_NO_REPEAT_NGRAM_SIZE = int(os.getenv("FINAL_NO_REPEAT_NGRAM_SIZE", "3"))
FINAL_NO_SPEECH_THRESHOLD = float(os.getenv("FINAL_NO_SPEECH_THRESHOLD", "0.5"))
FINAL_CHUNK_LENGTH = int(os.getenv("FINAL_CHUNK_LENGTH", "30"))
FINAL_CONDITION_ON_PREVIOUS_TEXT = os.getenv("FINAL_CONDITION_ON_PREVIOUS_TEXT", "false").lower() == "true"

app = FastAPI()
model = WhisperModel(
    MODEL_NAME,
    device=DEVICE,
    compute_type=COMPUTE_TYPE,
    cpu_threads=CPU_THREADS,
)
logger.info(
    f"Loaded Whisper model {MODEL_NAME} on {DEVICE} with compute_type={COMPUTE_TYPE}, cpu_threads={CPU_THREADS}, beam={FINAL_BEAM}, best_of={FINAL_BEST_OF}"
)


def pcm_to_numpy(audio_bytes: bytes) -> np.ndarray:
    return np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

@app.websocket("/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    audio_buffer = bytearray()
    language = "uk"
    final_sent = False

    try:
        init_data = await websocket.receive_json()
        if init_data.get("type") == "start":
            # Перевірка API ключа, якщо він налаштований на сервері
            client_key = init_data.get("api_key", "")
            if API_KEY and client_key != API_KEY:
                logger.warning(f"Unauthorized access attempt with key: {client_key}")
                await websocket.send_json({"type": "error", "message": "Unauthorized"})
                await websocket.close(code=1008)
                return

            language = init_data.get("language", "uk")
            
        while True:
            message = await websocket.receive()
            if "bytes" in message:
                audio_buffer.extend(message["bytes"])
            elif "text" in message:
                data = message["text"]
                try:
                    data = json.loads(data)
                except Exception:
                    continue

                if data.get("type") == "stop":
                    if final_sent:
                        logger.warning("Duplicate stop message ignored")
                        break

                    final_sent = True
                    try:
                        audio_np = pcm_to_numpy(bytes(audio_buffer)) if len(audio_buffer) > 0 else np.array([], dtype=np.float32)
                        segments, _ = await asyncio.to_thread(
                            model.transcribe,
                            audio_np,
                            language=language,
                            beam_size=FINAL_BEAM,
                            best_of=FINAL_BEST_OF,
                            patience=FINAL_PATIENCE,
                            temperature=[float(x) for x in FINAL_TEMPERATURE.split(",")],
                            repetition_penalty=FINAL_REPETITION_PENALTY,
                            no_repeat_ngram_size=FINAL_NO_REPEAT_NGRAM_SIZE,
                            no_speech_threshold=FINAL_NO_SPEECH_THRESHOLD,
                            chunk_length=FINAL_CHUNK_LENGTH,
                            condition_on_previous_text=FINAL_CONDITION_ON_PREVIOUS_TEXT,
                            vad_filter=True,
                            vad_parameters=dict(
                                threshold=0.5,
                                min_silence_duration_ms=500
                            ),
                        )
                        final_text = " ".join([s.text.strip() for s in segments if s.text.strip()])
                        logger.info(f"Sending final: {final_text}")
                        await websocket.send_json({"type": "final", "text": final_text})
                    except Exception as e:
                        logger.error(f"Final transcription error: {e}")
                        await websocket.send_json({"type": "final", "text": ""})
                    audio_buffer.clear()
                    break
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
