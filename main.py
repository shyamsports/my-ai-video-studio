
import os
import re
import shutil
import tempfile
import asyncio
import subprocess

import edge_tts
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.background import BackgroundTask
from pydantic import BaseModel, Field


app = FastAPI(title="My AI Voice Studio")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_WORDS = 10000
MAX_CHARACTERS = 200000
CHUNK_SIZE = 2500

VOICES = {
    "female": {
        "hi": "hi-IN-SwaraNeural",
        "en": "en-IN-NeerjaNeural",
    },
    "male": {
        "hi": "hi-IN-MadhurNeural",
        "en": "en-IN-PrabhatNeural",
    },
}

RATES = {
    "story": "-8%",
    "documentary": "-5%",
    "explainer": "+0%",
}


class VoiceRequest(BaseModel):
    script: str = Field(min_length=1, max_length=MAX_CHARACTERS)
    gender: str = "female"
    language: str = "hi"
    style: str = "story"


def split_text(text: str, max_size: int = CHUNK_SIZE):
    """Split long text into manageable chunks without losing text."""
    text = text.strip()
    if not text:
        return []

    sentences = re.split(r'(?<=[.!?।])\s+', text)
    chunks = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        while len(sentence) > max_size:
            available = max_size - len(current) - 1

            if current and available > 0:
                current += " " + sentence[:available]
                chunks.append(current.strip())
                current = ""
                sentence = sentence[available:].strip()
            else:
                if current:
                    chunks.append(current.strip())
                    current = ""
                chunks.append(sentence[:max_size].strip())
                sentence = sentence[max_size:].strip()

        if not sentence:
            continue

        if not current:
            current = sentence
        elif len(current) + len(sentence) + 1 <= max_size:
            current += " " + sentence
        else:
            chunks.append(current.strip())
            current = sentence

    if current:
        chunks.append(current.strip())

    return chunks


@app.get("/health")
async def health():
    return {"status": "ok", "service": "My AI Voice Studio"}


@app.post("/api/voiceover")
async def generate_voice(request: VoiceRequest):
    script = request.script.strip()

    if not script:
        raise HTTPException(status_code=400, detail="कृपया स्क्रिप्ट लिखें।")

    word_count = len(script.split())

    if word_count > MAX_WORDS:
        raise HTTPException(
            status_code=400,
            detail=f"अधिकतम {MAX_WORDS:,} शब्दों की अनुमति है। आपकी स्क्रिप्ट में {word_count:,} शब्द हैं।"
        )

    gender = request.gender.lower()
    language = request.language.lower()
    style = request.style.lower()

    if gender not in VOICES:
        raise HTTPException(status_code=400, detail="गलत voice selection.")

    if language not in ("hi", "en"):
        raise HTTPException(status_code=400, detail="गलत language selection.")

    if style not in RATES:
        raise HTTPException(status_code=400, detail="गलत voice style.")

    voice = VOICES[gender][language]
    rate = RATES[style]

    chunks = split_text(script)

    if not chunks:
        raise HTTPException(status_code=400, detail="स्क्रिप्ट खाली है।")

    temp_dir = tempfile.mkdtemp(prefix="voice_studio_")

    try:
        audio_files = []

        for index, chunk in enumerate(chunks):
            chunk_path = os.path.join(
                temp_dir,
                f"part_{index:05d}.mp3"
            )

            communicate = edge_tts.Communicate(
                chunk,
                voice,
                rate=rate
            )

            await communicate.save(chunk_path)
            audio_files.append(chunk_path)

        # Create FFmpeg concat list
        concat_file = os.path.join(temp_dir, "audio_list.txt")

        with open(concat_file, "w", encoding="utf-8") as file:
            for audio_file in audio_files:
                safe_path = audio_file.replace("'", "'\\''")
                file.write(f"file '{safe_path}'\n")

        output_path = os.path.join(temp_dir, "voiceover.mp3")

        # Join all generated audio chunks into one MP3
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", concat_file,
                "-vn",
                "-c:a", "libmp3lame",
                "-b:a", "128k",
                output_path
            ],
            capture_output=True,
            text=True,
            timeout=7200
        )

        if result.returncode != 0 or not os.path.exists(output_path):
            raise RuntimeError("ऑडियो फाइल जोड़ने में समस्या हुई।")

        return FileResponse(
            output_path,
            media_type="audio/mpeg",
            filename="voiceover.mp3",
            background=BackgroundTask(
                shutil.rmtree,
                temp_dir,
                ignore_errors=True
            )
        )

    except HTTPException:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    except Exception as error:
        shutil.rmtree(temp_dir, ignore_errors=True)
        print("VOICE GENERATION ERROR:", str(error))
        raise HTTPException(
            status_code=500,
            detail="Voice generate नहीं हो पाया। कृपया दोबारा कोशिश करें।"
        )
