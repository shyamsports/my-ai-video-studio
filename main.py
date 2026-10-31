
import os
import tempfile
import json
import asyncio

import edge_tts
from google import genai
from google.genai import types
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

app = FastAPI()


# ---------------------------------
# EXISTING VOICEOVER SYSTEM
# ---------------------------------

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


# ---------------------------------
# NEW AI SCENE PLANNER
# ---------------------------------

class SceneRequest(BaseModel):
    script: str


class Character(BaseModel):
    character_id: str
    description: str


class Scene(BaseModel):
    scene_number: int
    narration: str
    image_prompt: str
    character_ids: list[str]
    setting: str
    camera: str
    mood: str
    sound_effects: list[str]
    music_cue: str


class ScenePlan(BaseModel):
    story_summary: str
    visual_style: str
    characters: list[Character]
    scenes: list[Scene]


@app.post("/api/plan-scenes")
async def plan_scenes(data: SceneRequest):

    script = data.script.strip()

    if not script:
        raise HTTPException(400, "स्क्रिप्ट खाली है।")

    if len(script) > 50000:
        raise HTTPException(400, "स्क्रिप्ट बहुत लंबी है।")

    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        raise HTTPException(
            500,
            "Gemini API Key सेट नहीं है।"
        )

    prompt = f"""
You are an expert AI film director, storyboard artist,
screenwriter and visual continuity supervisor.

Create a detailed scene-by-scene visual plan from this script.

IMPORTANT RULES:

1. Divide the complete script into meaningful visual scenes.
2. Keep the original narration text in each scene.
3. Do not skip or invent narration.
4. Choose the visual style automatically according to the story.
   Choose an appropriate style such as 2D animation,
   3D animation, realistic cinematic or illustrated.
5. Keep every recurring character visually consistent.
6. Give every character a permanent character_id.
7. Repeat the same character appearance and clothing details
   in image prompts whenever that character appears.
8. Write image prompts in detailed English for image generators.
9. Describe setting, lighting, camera angle and composition.
10. Add suitable sound effects and background music cues.
11. Make scenes follow the story in the correct order.
12. Aim for approximately 5 to 10 seconds of visuals per scene
    where the narration allows it.
13. Do not add text, subtitles, logos or watermarks inside images
    unless the story specifically requires them.
14. Create no more than 30 scenes in this first planning pass.

SCRIPT:

{script}
"""

    client = None

    try:
        client = genai.Client(api_key=api_key)

        response = await asyncio.to_thread(
            client.models.generate_content,
            model="gemini-flash-latest",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_json_schema=ScenePlan.model_json_schema(),
                temperature=0.4,
                max_output_tokens=12000
            )
        )

        if not response.text:
            raise Exception("Gemini ने खाली जवाब दिया।")

        result = json.loads(response.text)

        plan = ScenePlan.model_validate(result)

        return plan.model_dump()

    except HTTPException:
        raise

    except Exception as error:
        print("Scene Planner Error:", str(error))

        raise HTTPException(
            500,
            "Scene Planner में समस्या आई। कृपया दोबारा कोशिश करें।"
        )

    finally:
        if client:
            client.close()
            
