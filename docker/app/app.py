from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import shutil
import os
import edge_tts
import base64
import requests

from model import get_tokenizer, get_device, get_image_transform, build_model, load_model_weights_from_path, generate_caption

app = FastAPI(title="DAT301m")

device = get_device()
tokenizer = get_tokenizer()
transform = get_image_transform()
model = build_model(tokenizer=tokenizer, device=device, max_length=30)
load_model_weights_from_path(model, "./weights/best_model_weights_v2.pth", device)

UPLOAD_DIR = "./uploads"
AUDIO_DIR = "./audio"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)


async def text_to_speech_base64(text: str, voice="vi-VN-HoaiMyNeural", rate="+30%"):
    """Tạo giọng nói tiếng Việt và trả về base64."""
    output_path = os.path.join(AUDIO_DIR, "temp_audio.mp3")
    tts = edge_tts.Communicate(text=text, voice=voice, rate=rate)

    with open(output_path, "wb") as f:
        async for chunk in tts.stream():
            if chunk["type"] == "audio":
                f.write(chunk["data"])

    with open(output_path, "rb") as f:
        audio_bytes = f.read()

    os.remove(output_path)
    return base64.b64encode(audio_bytes).decode("utf-8")



@app.post("/generate_caption")
async def caption_image(file: UploadFile = File(...), voice: str = Form("vi-VN-HoaiMyNeural"), rate: str = Form("+30%")):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    caption = generate_caption(model, file_path, tokenizer, transform, device, max_length=35)
    
    os.remove(file_path)

    if not caption:
        return JSONResponse({"error": "Failed to generate caption"}, status_code=500)

    try:
        audio_b64 = await text_to_speech_base64(caption, voice=voice, rate=rate)
    except Exception as e:
        return JSONResponse({"error": f"TTS failed: {e}"}, status_code=500)

    return JSONResponse({
        "caption": caption,
        "audio_base64": audio_b64,
        "voice": voice,
        "rate": rate
    })

@app.get("/")
async def root():
    return {"message": "ViTT5 Image Captioning API is running!"}
