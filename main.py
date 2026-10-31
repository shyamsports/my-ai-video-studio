import os
import tempfile

import edge_tts
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

app = FastAPI()

class VoiceRequest(BaseModel):
    script: str
    language: str = "hi"
    style: str = "story"

@app.get("/")
def home():
    return FileResponse("index.html")

@app.get("/health")
def health():
    return {"status": "Voice Studio is running"}

@app.post("/api/voiceover")
async def create_voiceover(data: VoiceRequest):

    script = data.script.strip()

    if not script:
        raise HTTPException(400, "स्क्रिप्ट खाली है।")

    if len(script) > 50000:
        raise HTTPException(400, "स्क्रिप्ट बहुत लंबी है।")

    voices = {
        "hi": "hi-IN-SwaraNeural",
        "en": "en-IN-NeerjaNeural"
    }

    rates = {
        "story": "-8%",
        "documentary": "-5%",
        "explainer": "+0%"
    }

    voice = voices.get(data.language)
    rate = rates.get(data.style)

    if not voice or not rate:
        raise HTTPException(400, "भाषा या Voice Style गलत है।")

    filename = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".mp3", delete=False
        ) as audio:
            filename = audio.name

        speaker = edge_tts.Communicate(
            text=script,
            voice=voice,
            rate=rate
        )

        await speaker.save(filename)

        with open(filename, "rb") as audio:
            result = audio.read()

        if not result:
            raise Exception("ऑडियो खाली मिला।")

        return Response(
            content=result,
            media_type="audio/mpeg"
        )

    except Exception as error:
        raise HTTPException(
            500,
            "आवाज़ बनाने में समस्या आई। दोबारा कोशिश करें।"
        )

    finally:
        if filename and os.path.exists(filename):
            os.remove(filename)
