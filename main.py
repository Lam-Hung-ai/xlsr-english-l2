import os
import random
import tempfile
import traceback
from contextlib import asynccontextmanager

import librosa
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from g2p_en import G2p
from transformers import AutoModelForCTC, AutoProcessor

# Đảm bảo bạn đã có file core.py hoặc thư mục core với __init__.py
try:
    from core import align_phonemes, arpabet2ipa
except ImportError:
    raise ImportError("Hãy kiểm tra lại cách bạn đặt file core.py hoặc thư mục core!")

MODEL_ID = "KoelLabs/xlsr-english-01"
SAMPLING_RATE = 16000
device = "cuda" if torch.cuda.is_available() else "cpu"

ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[*] Đang tải mô hình {MODEL_ID} lên {device}...")
    try:
        ml_models["processor"] = AutoProcessor.from_pretrained(MODEL_ID)
        ml_models["model"] = AutoModelForCTC.from_pretrained(MODEL_ID).to(device)
        ml_models["g2p"] = G2p()

        # Kiểm tra file sentences.txt an toàn
        import os

        if os.path.exists("data/sentences.txt"):
            with open("data/sentences.txt", encoding="utf-8") as f:
                ml_models["sentences"] = [line.strip() for line in f if line.strip()]
        else:
            print(
                "[!] Cảnh báo: data/sentences.txt không tồn tại, dùng dữ liệu mặc định."
            )
            ml_models["sentences"] = ["this is a test sentence"]

        print("[+] Model đã sẵn sàng!")
    except Exception as e:
        print(f"[!] Lỗi khi load model: {e}")
        traceback.print_exc()

    yield
    ml_models.clear()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def get_index():
    try:
        with open("index.html", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>Error: index.html not found</h1>"


@app.get("/get-sample")
async def get_sample():
    if not ml_models.get("sentences"):
        return {"text": "Error", "ipa_target": ""}
    text = random.choice(ml_models["sentences"])
    arpabet_list = ml_models["g2p"](text)
    ipa_target = arpabet2ipa(" ".join(arpabet_list))
    return {"text": text, "ipa_target": ipa_target}


@app.post("/analyze")
async def analyze(audio_file: UploadFile = File(...), target_ipa: str = Form(...)):
    # 1. Tạo file tạm an toàn (Dùng mkstemp để tránh lỗi truy cập file trên Windows)
    fd, temp_path = tempfile.mkstemp(suffix=".webm")
    try:
        with os.fdopen(fd, 'wb') as tmp:
            tmp.write(await audio_file.read())

        # 2. Đọc audio và xử lý nhiễu
        y, sr = librosa.load(temp_path, sr=SAMPLING_RATE)
        if len(y) == 0:
            raise HTTPException(status_code=400, detail="Audio trống hoặc không thể đọc được")
            
        y, _ = librosa.effects.trim(y) # Bỏ khoảng lặng đầu/cuối

        # 3. AI Inference
        inputs = ml_models["processor"](y, sampling_rate=SAMPLING_RATE, return_tensors="pt")
        input_values = inputs.input_values.to(device)

        with torch.no_grad():
            logits = ml_models["model"](input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        # Decode ra text: "hello"
        predicted_text = ml_models["processor"].batch_decode(predicted_ids)[0].lower()

        # 4. CHỐT CHẶN QUAN TRỌNG: Chuyển Text nhận diện được sang IPA
        # Nếu model của bạn output thẳng IPA thì bỏ qua bước này. 
        # Nhưng đa số model XLSR English output ra text.
        user_phonemes_list = ml_models["g2p"](predicted_text)
        user_ipa = arpabet2ipa(" ".join(user_phonemes_list))

        # 5. MDD Logic
        # Làm sạch chuỗi target (bỏ dấu / và khoảng trắng thừa)
        clean_target = target_ipa.replace("/", "").strip()
        
        # results nên là list các tuple: (target, user, status)
        results = align_phonemes(clean_target, user_ipa)

        # 6. Tính điểm dựa trên số âm vị đúng
        target_tokens = [t for t, u, s in results if t != "-"]
        correct_count = sum(1 for t, u, s in results if s == "Correct")
        
        total_phonemes = len(target_tokens)
        score = (correct_count / total_phonemes * 100) if total_phonemes > 0 else 0

        return {
            "score": round(score, 2),
            "target_ipa": clean_target,
            "user_ipa": user_ipa,
            "recognized_text": predicted_text,
            "diagnosis": [{"t": t, "u": u, "s": s} for t, u, s in results],
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
