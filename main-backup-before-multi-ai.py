
import os
import tempfile
import json
import asyncio
import urllib.request
import urllib.error

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

    except Exception:
        raise HTTPException(
            500,
            "आवाज़ बनाने में समस्या आई। दोबारा कोशिश करें।"
        )

    finally:
        if filename and os.path.exists(filename):
            os.remove(filename)


# ---------------------------------
# AI SCENE PLANNER
# ---------------------------------

class SceneRequest(BaseModel):
    script: str
    video_format: str


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
    video_format: str
    aspect_ratio: str
    story_summary: str
    visual_style: str
    characters: list[Character]
    scenes: list[Scene]


# ---------------------------------
# SHARED PROMPT
# ---------------------------------

def create_scene_prompt(script, video_format, aspect_ratio,
                        format_name, framing_rules):

    return f"""
You are an expert AI film director, storyboard artist,
screenwriter and visual continuity supervisor.

Create a detailed scene-by-scene visual plan from this script.

VIDEO FORMAT: {format_name}
ASPECT RATIO: {aspect_ratio}

FORMAT-SPECIFIC COMPOSITION RULES:
{framing_rules}

IMPORTANT RULES:

1. Divide the complete script into meaningful visual scenes.
2. Keep the original narration text in each scene.
3. Do not skip or invent narration.
4. Choose visual style automatically according to the story:
   2D animation, 3D animation, realistic cinematic or illustrated.
5. Keep recurring characters visually consistent.
6. Give every character a permanent character_id.
7. Repeat the same character appearance and clothing details
   in image prompts whenever that character appears.
8. Write detailed image prompts in English.
9. Describe setting, lighting, camera angle and composition.
10. Add suitable sound effects and background music cues.
11. Keep scenes in the correct story order.
12. Aim for approximately 5 to 10 seconds per scene.
13. Do not add text, subtitles, logos or watermarks inside images.
14. Create no more than 30 scenes.
15. Every image prompt must follow the selected aspect ratio.
16. Do not change the selected video format.

Return ONLY valid JSON with these exact fields:

{{
  "video_format": "{format_name}",
  "aspect_ratio": "{aspect_ratio}",
  "story_summary": "Short story summary",
  "visual_style": "Selected visual style",
  "characters": [
    {{
      "character_id": "character_1",
      "description": "Detailed appearance"
    }}
  ],
  "scenes": [
    {{
      "scene_number": 1,
      "narration": "Original narration",
      "image_prompt": "Detailed English image prompt",
      "character_ids": ["character_1"],
      "setting": "Location and environment",
      "camera": "Camera angle and movement",
      "mood": "Scene mood",
      "sound_effects": ["Sound effect"],
      "music_cue": "Background music description"
    }}
  ]
}}

SCRIPT:
{script}
"""


# ---------------------------------
# CLOUDFLARE BACKUP
# ---------------------------------

def generate_with_cloudflare(prompt):

    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    api_token = os.environ.get("CLOUDFLARE_API_TOKEN")

    if not account_id or not api_token:
        raise Exception("Cloudflare credentials missing")

    model = "@cf/zai-org/glm-4.7-flash"

    url = (
        "https://api.cloudflare.com/client/v4/accounts/"
        + account_id
        + "/ai/run/"
        + model
    )

    payload = {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an expert film director. "
                    "Return only valid JSON. No markdown."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "response_format": {
            "type": "json_object"
        },
        "temperature": 0.3,
        "max_tokens": 12000
    }

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json"
        },
        method="POST"
    )

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            result = json.loads(response.read().decode("utf-8"))

    except urllib.error.HTTPError as error:
        error_body = error.read().decode("utf-8", errors="replace")
        raise Exception(
            f"Cloudflare HTTP {error.code}: {error_body[:1000]}"
        )

    if not result.get("success"):
        raise Exception(
            "Cloudflare API error: " +
            str(result.get("errors", []))
        )

    output = result.get("result", {}).get("response")

    if isinstance(output, dict):
        return output

    if not output:
        raise Exception("Cloudflare ने खाली जवाब दिया।")

    if isinstance(output, str):
        output = output.strip()

        if output.startswith("```"):
            output = output.replace("```json", "", 1)
            output = output.replace("```", "").strip()

        return json.loads(output)

    raise Exception("Cloudflare का जवाब समझ नहीं आया।")


# ---------------------------------
# GEMINI + CLOUDFLARE AUTO FALLBACK
# ---------------------------------

@app.post("/api/plan-scenes")
async def plan_scenes(data: SceneRequest):

    script = data.script.strip()

    if not script:
        raise HTTPException(400, "स्क्रिप्ट खाली है।")

    if len(script) > 50000:
        raise HTTPException(400, "स्क्रिप्ट बहुत लंबी है।")

    if data.video_format not in ["landscape", "portrait"]:
        raise HTTPException(
            400,
            "कृपया सही Video Format चुनें।"
        )

    if data.video_format == "landscape":

        aspect_ratio = "16:9"
        format_name = "Full YouTube Video"

        framing_rules = """
- Compose every image specifically for a wide 16:9 landscape frame.
- Use horizontal compositions and cinematic wide shots.
- Keep important characters and story details inside the frame.
- Use horizontal space to show the environment clearly.
"""

    else:

        aspect_ratio = "9:16"
        format_name = "YouTube Shorts / Reels"

        framing_rules = """
- Compose every image specifically for a vertical 9:16 portrait frame.
- Use vertical compositions and portrait-friendly camera framing.
- Keep important characters and faces inside the central safe area.
- Avoid placing important actions near extreme left and right edges.
- Use foreground, middle ground and background effectively.
- Do not simply crop a landscape composition into a vertical image.
"""

    prompt = create_scene_prompt(
        script,
        data.video_format,
        aspect_ratio,
        format_name,
        framing_rules
    )

    # FIRST PROVIDER: GEMINI

    api_key = os.environ.get("GEMINI_API_KEY")

    if api_key:

        client = None

        try:
            print("Scene Planner: Trying Gemini...")

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

            print("Scene Planner: Gemini succeeded")

            return plan.model_dump()

        except Exception as error:
            print("Gemini failed:", str(error))

        finally:
            if client:
                client.close()

    else:
        print("Gemini key missing. Trying Cloudflare...")


    # SECOND PROVIDER: CLOUDFLARE

    try:
        print("Scene Planner: Trying Cloudflare backup...")

        result = await asyncio.to_thread(
            generate_with_cloudflare,
            prompt
        )

        result["video_format"] = format_name
        result["aspect_ratio"] = aspect_ratio

        plan = ScenePlan.model_validate(result)

        print("Scene Planner: Cloudflare succeeded")

        return plan.model_dump()

    except Exception as error:

        print("Cloudflare failed:", str(error))

        raise HTTPException(
            503,
            "Gemini और Cloudflare दोनों अभी जवाब नहीं दे पाए। "
            "कृपया कुछ देर बाद दोबारा कोशिश करें।"
        )
