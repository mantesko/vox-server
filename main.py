import os
import json
import asyncio
import logging
import numpy as np
from fastapi import FastAPI, WebSocket
from faster_whisper import WhisperModel
from groq import Groq

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
FINAL_NO_SPEECH_THRESHOLD = float(os.getenv("FINAL_NO_SPEECH_THRESHOLD", "0.6"))
FINAL_CHUNK_LENGTH = int(os.getenv("FINAL_CHUNK_LENGTH", "15"))
FINAL_CONDITION_ON_PREVIOUS_TEXT = os.getenv("FINAL_CONDITION_ON_PREVIOUS_TEXT", "false").lower() == "true"
VAD_THRESHOLD = float(os.getenv("VAD_THRESHOLD", "0.35"))
VAD_MIN_SILENCE_MS = int(os.getenv("VAD_MIN_SILENCE_MS", "300"))
LOG_PROB_THRESHOLD = float(os.getenv("LOG_PROB_THRESHOLD", "-1.0"))
COMPRESSION_RATIO_THRESHOLD = float(os.getenv("COMPRESSION_RATIO_THRESHOLD", "2.4"))
INITIAL_PROMPT = os.getenv("INITIAL_PROMPT", "")

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


async def post_process_text(raw_text: str, groq_api_key: str = None, groq_prompt: str = None,
                            groq_model: str = None, groq_temperature: float = None) -> str:
    if not raw_text.strip() or not groq_api_key:
        return raw_text

    try:
        client = Groq(api_key=groq_api_key)
        completion = await asyncio.to_thread(
            client.chat.completions.create,
            model=groq_model or "llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": groq_prompt},
                {"role": "user", "content": raw_text},
            ],
            temperature=groq_temperature if groq_temperature is not None else 0.2,
        )
        result = completion.choices[0].message.content.strip()
        logger.info(f"Groq post-processing: {len(raw_text)} -> {len(result)} chars")
        return result
    except Exception as e:
        logger.warning(f"Groq post-processing failed, returning raw text: {e}")
        return raw_text

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
            client_prompt = init_data.get("initial_prompt") or ""
            client_groq_key = init_data.get("groq_api_key") or ""
            client_groq_prompt = init_data.get("groq_prompt") or ""
            client_groq_model = init_data.get("groq_model") or ""
            client_groq_temperature = init_data.get("groq_temperature")
            if client_groq_temperature is not None:
                client_groq_temperature = float(client_groq_temperature)
            logger.info(f"Client prompt: '{client_prompt[:80]}...' " if client_prompt else "Client prompt: (empty)")
            if client_groq_key:
                logger.info("Client Groq key received")
            
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
                            initial_prompt=client_prompt or INITIAL_PROMPT or None,
                            vad_filter=True,
                            vad_parameters=dict(
                                threshold=VAD_THRESHOLD,
                                min_silence_duration_ms=VAD_MIN_SILENCE_MS
                            ),
                            log_prob_threshold=LOG_PROB_THRESHOLD,
                            compression_ratio_threshold=COMPRESSION_RATIO_THRESHOLD,
                        )
                        final_text = " ".join([s.text.strip() for s in segments if s.text.strip()])
                        logger.info(f"Whisper raw: {final_text}")
                        final_text = await post_process_text(final_text, client_groq_key, client_groq_prompt,
                                                             client_groq_model, client_groq_temperature)
                        logger.info(f"Sending final: {final_text}")
                        await websocket.send_json({"type": "final", "text": final_text})
                    except Exception as e:
                        logger.error(f"Final transcription error: {e}")
                        await websocket.send_json({"type": "final", "text": ""})
                    audio_buffer.clear()
                    break
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
