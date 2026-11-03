
# 🎙️ My AI Voice Studio

**My AI Voice Studio** is a web-based AI voice generation application that converts written scripts into speech using Microsoft Edge TTS.

Generate Hindi and Indian English voiceovers with Male and Female AI voices. Listen to your generated audio directly on the website and download it as an MP3 file.

🌐 **Live Website:** https://my-ai-video-studio.onrender.com

---

## ✨ Features

- 🎤 Male and Female AI voice selection
- 🇮🇳 Hindi voice generation
- 🇮🇳 Indian English voice generation
- 📝 Script support up to 10,000 words
- 📊 Live word counter
- 📖 Story, Documentary and Explainer voice styles
- 🔊 Listen to generated audio directly on the website
- 📥 Download generated audio as MP3
- 📱 Responsive web interface
- 💸 No paid API key required for Edge TTS

---

## 🎙️ Available AI Voices

### Female Voices

| Language | Voice |
|---|---|
| Hindi | hi-IN-SwaraNeural |
| English (India) | en-IN-NeerjaNeural |

### Male Voices

| Language | Voice |
|---|---|
| Hindi | hi-IN-MadhurNeural |
| English (India) | en-IN-PrabhatNeural |

---

## 🎚️ Voice Styles

| Style | Description |
|---|---|
| Story | Suitable for storytelling and narration |
| Documentary | Suitable for documentary narration |
| Explainer | Suitable for educational and informational content |

---

## 🛠️ Technologies Used

- Python 3.11
- FastAPI
- Microsoft Edge TTS
- FFmpeg
- HTML
- CSS
- JavaScript
- Docker
- Render

---

## 🚀 How to Use

1. Open My AI Voice Studio.
2. Enter or paste your script.
3. Select Male or Female voice.
4. Choose Hindi or English.
5. Select a voice style.
6. Click **Generate Voice**.
7. Play the generated audio.
8. Download your MP3 file.

---

## 💻 Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/shyamsports/my-ai-video-studio.git
```

### 2. Open the Project Folder

```bash
cd my-ai-video-studio
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install FFmpeg

Install FFmpeg on your system and make sure it is available in your system PATH.

### 5. Start the Application

```bash
uvicorn main:app --host 0.0.0.0 --port 10000
```

### 6. Open in Your Browser

```text
http://localhost:10000
```

---

## 📡 API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Checks application health |
| `/api/voiceover` | POST | Generates an AI voiceover |

---

## ⚙️ Important Information

- An active internet connection is required for Edge TTS.
- Voice generation time depends on script length and server resources.
- Long scripts may take more time to generate.
- The 10,000-word limit does not guarantee that every long request will complete within the hosting provider's request timeout.
- Availability depends on the Edge TTS service and hosting environment.

---

## 👨‍💻 Project Information

**Project Name:** My AI Voice Studio

**GitHub Repository:**  
https://github.com/shyamsports/my-ai-video-studio

**Live Website:**  
https://my-ai-video-studio.onrender.com

---

⭐ If you find this project useful, consider giving the repository a star!

