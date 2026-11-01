import os
import re
import json
import asyncio
import tempfile
import urllib.request
import urllib.error

import edge_tts
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

app = FastAPI()


# ---------------------------------
# VOICEOVER SYSTEM - KEEP AS IT IS
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

    voices = {"hi": "hi-IN-SwaraNeural", "en": "en-IN-NeerjaNeural"}
    rates = {"story": "-8%", "documentary": "-5%", "explainer": "+0%"}
    voice = voices.get(data.language)
    rate = rates.get(data.style)
    if not voice or not rate:
        raise HTTPException(400, "भाषा या Voice Style गलत है।")

    filename = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as audio:
            filename = audio.name
        speaker = edge_tts.Communicate(text=script, voice=voice, rate=rate)
        await speaker.save(filename)
        with open(filename, "rb") as audio:
            result = audio.read()
        if not result:
            raise Exception("ऑडियो खाली मिला।")
        return Response(content=result, media_type="audio/mpeg")
    except Exception:
        raise HTTPException(500, "आवाज़ बनाने में समस्या आई। दोबारा कोशिश करें।")
    finally:
        if filename and os.path.exists(filename):
            os.remove(filename)


# ---------------------------------
# SCENE PLANNER DATA MODELS
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
# SCRIPT CHUNKING
# ---------------------------------

def split_script(script: str, sentences_per_chunk: int = 8):
    """Split by sentence while preserving the original sentence wording."""
    parts = re.split(r"(?<=[.!?।])\s+", script.strip())
    sentences = [part.strip() for part in parts if part.strip()]
    if not sentences:
        sentences = [script.strip()]
    return [sentences[i:i + sentences_per_chunk]
            for i in range(0, len(sentences), sentences_per_chunk)]


def extract_json(text: str):
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise ValueError("AI ने valid JSON नहीं दिया।")


def make_prompt(sentences, format_name, aspect_ratio, framing_rules, known_characters, visual_style):
    numbered = "\n".join(f'{i + 1}. {sentence}' for i, sentence in enumerate(sentences))
    known = json.dumps(known_characters, ensure_ascii=False)
    return f"""You are an expert film director and storyboard artist.
Create exactly ONE visual scene for EACH numbered narration segment below.
Do not combine, omit, reorder, rewrite, or invent narration. The backend will insert the original narration text itself.

VIDEO FORMAT: {format_name}
ASPECT RATIO: {aspect_ratio}
VISUAL STYLE: {visual_style or 'Choose the most suitable style: 2D animation, 3D animation, realistic cinematic, or illustration.'}
FRAMING RULES:
{framing_rules}

CHARACTER CONTINUITY:
Existing character bible (reuse the same IDs and descriptions when applicable): {known}
Do not create a new ID for an existing character. If no recurring character is visible, use an empty character_ids array.

For every scene provide:
- image_prompt: detailed English prompt, with character appearance, action, environment, lighting and composition; no text/logo/watermark.
- character_ids: list of IDs from the character bible or new IDs defined in characters.
- setting, camera, mood, sound_effects (array), music_cue.
- Add new recurring characters to characters with a stable ID and detailed appearance.

Return ONLY valid JSON in this exact shape:
{{
  "story_summary": "Short summary of the story so far",
  "visual_style": "Chosen visual style",
  "characters": [{{"character_id": "character_1", "description": "Detailed consistent appearance"}}],
  "scenes": [
    {{"segment_number": 1, "image_prompt": "Detailed English image prompt", "character_ids": [], "setting": "...", "camera": "...", "mood": "...", "sound_effects": ["..."], "music_cue": "..."}}
  ]
}}

Create exactly {len(sentences)} scene objects. segment_number must match the numbered narration segment.
NARRATION SEGMENTS:
{numbered}
"""


# ---------------------------------
# AI PROVIDERS: MISTRAL -> GROQ -> CLOUDFLARE
# ---------------------------------

def post_json(url, payload, headers, timeout=100):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {detail[:500]}")


def generate_mistral(prompt):
    key = os.getenv("MISTRAL_API_KEY")
    if not key:
        raise RuntimeError("MISTRAL_API_KEY missing")
    result = post_json(
        "https://api.mistral.ai/v1/chat/completions",
        {
            "model": os.getenv("MISTRAL_MODEL", "mistral-small-latest"),
            "messages": [
                {"role": "system", "content": "Return only valid JSON. No markdown."},
                {"role": "user", "content": prompt}
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.3,
            "max_tokens": 7000
        },
        {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    )
    return result["choices"][0]["message"]["content"]


def generate_groq(prompt):
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY missing")
    result = post_json(
        "https://api.groq.com/openai/v1/chat/completions",
        {
            "model": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            "messages": [
                {"role": "system", "content": "Return only valid JSON. No markdown."},
                {"role": "user", "content": prompt}
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.3,
            "max_tokens": 7000
        },
        {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    )
    return result["choices"][0]["message"]["content"]


def generate_cloudflare(prompt):
    account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
    token = os.getenv("CLOUDFLARE_API_TOKEN")
    if not account_id or not token:
        raise RuntimeError("Cloudflare credentials missing")
    model = os.getenv("CLOUDFLARE_MODEL", "@cf/meta/llama-3.1-8b-instruct")
    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}"
    result = post_json(
        url,
        {"prompt": "Return only valid JSON. No markdown.\n\n" + prompt,
         "temperature": 0.3, "max_tokens": 7000},
        {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    )
    if not result.get("success", False):
        raise RuntimeError("Cloudflare API returned an unsuccessful response")
    output = result.get("result", {}).get("response")
    if isinstance(output, dict):
        return json.dumps(output, ensure_ascii=False)
    if not output:
        raise RuntimeError("Cloudflare returned an empty response")
    return output


def generate_with_fallback(prompt):
    providers = [
        ("Mistral", generate_mistral),
        ("Groq", generate_groq),
        ("Cloudflare", generate_cloudflare)
    ]
    errors = []
    for name, generator in providers:
        try:
            print(f"Scene Planner: trying {name}")
            raw = generator(prompt)
            parsed = extract_json(raw)
            if not isinstance(parsed, dict):
                raise ValueError("AI response is not a JSON object")
            print(f"Scene Planner: {name} succeeded")
            return parsed
        except Exception as error:
            message = str(error)
            print(f"Scene Planner: {name} failed: {message[:300]}")
            errors.append(f"{name}: {message[:180]}")
    raise RuntimeError("All AI providers failed. " + " | ".join(errors))


# ---------------------------------
# SCENE PLANNER API
# ---------------------------------

@app.post("/api/plan-scenes")
async def plan_scenes(data: SceneRequest):
    script = data.script.strip()
    if not script:
        raise HTTPException(400, "स्क्रिप्ट खाली है।")
    if len(script) > 50000:
        raise HTTPException(400, "स्क्रिप्ट बहुत लंबी है।")
    if data.video_format not in ["landscape", "portrait"]:
        raise HTTPException(400, "कृपया सही Video Format चुनें।")

    if data.video_format == "landscape":
        aspect_ratio = "16:9"
        format_name = "Full YouTube Video"
        framing_rules = "Use a wide horizontal 16:9 composition, cinematic wide shots, and keep important subjects comfortably inside frame."
    else:
        aspect_ratio = "9:16"
        format_name = "Vertical Video"
        framing_rules = "Use a vertical 9:16 portrait composition designed natively for vertical viewing; keep faces and key action in the central safe area. Do not crop a landscape composition."

    chunks = split_script(script)
    all_characters = {}
    all_scenes = []
    story_summary = ""
    visual_style = ""

    try:
        for chunk_index, sentences in enumerate(chunks):
            prompt = make_prompt(
                sentences, format_name, aspect_ratio, framing_rules,
                list(all_characters.values()), visual_style
            )
            result = await asyncio.to_thread(generate_with_fallback, prompt)

            if not isinstance(result.get("scenes"), list):
                raise ValueError("AI response में scenes की list नहीं मिली।")
            if len(result["scenes"]) != len(sentences):
                raise ValueError(
                    f"Chunk {chunk_index + 1}: AI ने {len(result['scenes'])} scenes दिए, "
                    f"जबकि {len(sentences)} चाहिए थे। कृपया दोबारा कोशिश करें।"
                )

            for character in result.get("characters", []):
                if isinstance(character, dict) and character.get("character_id"):
                    all_characters[character["character_id"]] = {
                        "character_id": character["character_id"],
                        "description": str(character.get("description", ""))
                    }

            if result.get("visual_style") and not visual_style:
                visual_style = str(result["visual_style"])
            if result.get("story_summary"):
                story_summary = str(result["story_summary"])

            ordered = sorted(
                result["scenes"],
                key=lambda item: int(item.get("segment_number", 0))
            )
            for index, item in enumerate(ordered):
                if not isinstance(item, dict):
                    raise ValueError("AI scene का format गलत है।")
                # Use the exact source sentence, not AI-rewritten narration.
                all_scenes.append({
                    "scene_number": len(all_scenes) + 1,
                    "narration": sentences[index],
                    "image_prompt": str(item.get("image_prompt", "Create a suitable cinematic scene.")),
                    "character_ids": item.get("character_ids", []) if isinstance(item.get("character_ids", []), list) else [],
                    "setting": str(item.get("setting", "")),
                    "camera": str(item.get("camera", "")),
                    "mood": str(item.get("mood", "")),
                    "sound_effects": item.get("sound_effects", []) if isinstance(item.get("sound_effects", []), list) else [],
                    "music_cue": str(item.get("music_cue", ""))
                })

        plan = ScenePlan(
            video_format=format_name,
            aspect_ratio=aspect_ratio,
            story_summary=story_summary or "Scene plan generated from the supplied script.",
            visual_style=visual_style or "Cinematic animation / illustration",
            characters=[Character(**item) for item in all_characters.values()],
            scenes=[Scene(**item) for item in all_scenes]
        )
        return plan.model_dump()

    except HTTPException:
        raise
    except Exception as error:
        print("Scene Planner final error:", str(error)[:1000])
        raise HTTPException(
            503,
            "Scene Planner अभी पूरा प्लान नहीं बना पाया। Provider की free limit, network या AI response में समस्या हो सकती है। कुछ देर बाद छोटी स्क्रिप्ट से दोबारा कोशिश करें।")
