
import os
import re
import shutil
import tempfile
import subprocess
import logging

import edge_tts

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.background import BackgroundTask
from pydantic import BaseModel, Field


# --------------------------------------------------
# APP CONFIGURATION
# --------------------------------------------------

app = FastAPI(title="My AI Voice Studio")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voice_studio")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# LIMITS
# --------------------------------------------------

MAX_WORDS = 10000
MAX_CHARACTERS = 200000

# Text is divided into smaller pieces for voice generation.
CHUNK_SIZE = 2500


# --------------------------------------------------
# AVAILABLE VOICES
# --------------------------------------------------

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


# --------------------------------------------------
# SPEAKING STYLES
# --------------------------------------------------

RATES = {
    "story": "-8%",
    "documentary": "-5%",
    "explainer": "+0%",
}


# --------------------------------------------------
# REQUEST MODEL
# --------------------------------------------------

class VoiceRequest(BaseModel):
    script: str = Field(
        min_length=1,
        max_length=MAX_CHARACTERS
    )

    gender: str = "female"
    language: str = "hi"
    style: str = "story"


# --------------------------------------------------
# TEXT CHUNKING
# --------------------------------------------------

def split_text(text: str, max_size: int = CHUNK_SIZE):
    """
    Split long scripts into manageable pieces.

    - Tries to split at sentence boundaries.
    - Splits very long sentences at word boundaries.
    - Handles unusually long individual words.
    - Does not intentionally discard text.
    """

    text = text.strip()

    if not text:
        return []

    sentences = re.split(
        r'(?<=[.!?।])\s+',
        text
    )

    pieces = []

    for sentence in sentences:
        sentence = sentence.strip()

        if not sentence:
            continue

        if len(sentence) <= max_size:
            pieces.append(sentence)
            continue

        words = sentence.split()
        current_piece = ""

        for word in words:

            # Handle an unusually long word.
            if len(word) > max_size:

                if current_piece:
                    pieces.append(current_piece)
                    current_piece = ""

                for start in range(0, len(word), max_size):
                    pieces.append(
                        word[start:start + max_size]
                    )

                continue

            candidate = (
                word if not current_piece
                else current_piece + " " + word
            )

            if len(candidate) <= max_size:
                current_piece = candidate

            else:
                if current_piece:
                    pieces.append(current_piece)

                current_piece = word

        if current_piece:
            pieces.append(current_piece)

    # Pack smaller pieces together without exceeding the limit.
    chunks = []
    current_chunk = ""

    for piece in pieces:

        candidate = (
            piece if not current_chunk
            else current_chunk + " " + piece
        )

        if len(candidate) <= max_size:
            current_chunk = candidate

        else:
            if current_chunk:
                chunks.append(current_chunk)

            current_chunk = piece

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/")
async def home():
    return FileResponse(
        path="index.html",
        media_type="text/html"
    )

@app.get("/health")
async def health():

    return {
        "status": "ok",
        "service": "My AI Voice Studio"
    }


# --------------------------------------------------
# VOICE GENERATION
# --------------------------------------------------

@app.post("/api/voiceover")
async def generate_voice(request: VoiceRequest):

    script = request.script.strip()

    # Empty script check
    if not script:
        raise HTTPException(
            status_code=400,
            detail="कृपया अपनी स्क्रिप्ट लिखें।"
        )

    # Word count check
    word_count = len(script.split())

    if word_count > MAX_WORDS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"अधिकतम {MAX_WORDS:,} शब्दों की अनुमति है। "
                f"आपकी स्क्रिप्ट में {word_count:,} शब्द हैं।"
            )
        )

    # Character count check
    if len(script) > MAX_CHARACTERS:
        raise HTTPException(
            status_code=400,
            detail="स्क्रिप्ट character limit से बड़ी है।"
        )

    # Normalize selections
    gender = request.gender.strip().lower()
    language = request.language.strip().lower()
    style = request.style.strip().lower()

    # Validate voice selection
    if gender not in VOICES:
        raise HTTPException(
            status_code=400,
            detail="गलत voice selection."
        )

    if language not in ("hi", "en"):
        raise HTTPException(
            status_code=400,
            detail="गलत language selection."
        )

    if style not in RATES:
        raise HTTPException(
            status_code=400,
            detail="गलत speaking style."
        )

    voice = VOICES[gender][language]
    rate = RATES[style]

    # Divide script into chunks
    chunks = split_text(script)

    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="स्क्रिप्ट खाली है।"
        )

    logger.info(
        "Voice generation started | words=%s | chunks=%s | voice=%s | style=%s",
        word_count,
        len(chunks),
        voice,
        style
    )

    # Temporary directory for audio files
    temp_dir = tempfile.mkdtemp(
        prefix="voice_studio_"
    )

    try:

        audio_files = []

        # Generate every audio chunk
        for index, chunk in enumerate(chunks):

            chunk_path = os.path.join(
                temp_dir,
                f"part_{index:05d}.mp3"
            )

            communicate = edge_tts.Communicate(
                text=chunk,
                voice=voice,
                rate=rate
            )

            await communicate.save(chunk_path)

            if (
                not os.path.exists(chunk_path)
                or os.path.getsize(chunk_path) == 0
            ):
                raise RuntimeError(
                    f"Audio chunk {index + 1} was empty."
                )

            audio_files.append(chunk_path)

            logger.info(
                "Generated audio chunk %s/%s",
                index + 1,
                len(chunks)
            )

        # --------------------------------------------------
        # CREATE FFMPEG CONCAT FILE
        # --------------------------------------------------

        concat_file = os.path.join(
            temp_dir,
            "audio_list.txt"
        )

        with open(
            concat_file,
            "w",
            encoding="utf-8"
        ) as file:

            for audio_file in audio_files:

                safe_path = audio_file.replace(
                    "'",
                    "'\\''"
                )

                file.write(
                    f"file '{safe_path}'\n"
                )

        # --------------------------------------------------
        # MERGE AUDIO CHUNKS
        # --------------------------------------------------

        output_path = os.path.join(
            temp_dir,
            "voiceover.mp3"
        )

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

        if result.returncode != 0:

            logger.error(
                "FFmpeg error: %s",
                result.stderr[-4000:]
            )

            raise RuntimeError(
                "Audio files को जोड़ने में समस्या हुई।"
            )

        if (
            not os.path.exists(output_path)
            or os.path.getsize(output_path) == 0
        ):
            raise RuntimeError(
                "Final audio file नहीं बन पाई।"
            )

        logger.info(
            "Voice generation completed | words=%s | size=%s bytes",
            word_count,
            os.path.getsize(output_path)
        )

        # --------------------------------------------------
        # SEND MP3 TO FRONTEND
        # --------------------------------------------------

        return FileResponse(
            path=output_path,
            media_type="audio/mpeg",
            filename="my-ai-voice.mp3",
            background=BackgroundTask(
                shutil.rmtree,
                temp_dir,
                ignore_errors=True
            )
        )

    except HTTPException:

        shutil.rmtree(
            temp_dir,
            ignore_errors=True
        )

        raise

    except subprocess.TimeoutExpired:

        logger.exception(
            "FFmpeg processing timed out."
        )

        shutil.rmtree(
            temp_dir,
            ignore_errors=True
        )

        raise HTTPException(
            status_code=500,
            detail="Audio processing में बहुत समय लग गया। कृपया दोबारा कोशिश करें।"
        )

    except Exception as error:

        logger.exception(
            "Voice generation failed: %s",
            str(error)
        )

        shutil.rmtree(
            temp_dir,
            ignore_errors=True
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Voice generate नहीं हो पाई। "
                "Internet connection जाँचें और थोड़ी देर बाद दोबारा कोशिश करें।"
            )
        )
