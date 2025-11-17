import os
import json
import re
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from googletrans import Translator
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
# --- Configuration ---
MODEL_PATH = "./eng_spam_model"
MAX_LENGTH = 64

# --- API Setup ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan handler: load model and tokenizer before the app starts."""
    global model, tokenizer, ID2LABEL

    try:
        # 1. Load label mappings
        with open(f"{MODEL_PATH}/label_mappings.json", "r", encoding="utf-8") as f:
            mappings = json.load(f)
        ID2LABEL = {int(k): v for k, v in mappings["id2label"].items()}
        LABEL2ID = mappings["label2id"]
        NUM_LABELS = len(ID2LABEL)

        # 2. Load tokenizer and model
        tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
        model = DistilBertForSequenceClassification.from_pretrained(
            MODEL_PATH,
            num_labels=NUM_LABELS,
            id2label=ID2LABEL,
            label2id=LABEL2ID
        )
        model.eval()  # Set to evaluation mode
        print("✅ API: Model and Resources loaded successfully.")

        yield

    except Exception as e:
        print(f"❌ API Error: Failed to load model from {MODEL_PATH}.")
        print(f"Details: {e}")
        # Re-raise the error to prevent the server from starting
        raise RuntimeError("Failed to load ML model dependencies.")


app = FastAPI(
    title="MLPM Spam Detector API",
    description="A service to classify messages as spam, scam, toxic, or ham.",
    version="1.0.0",
    lifespan=lifespan,
)

# --- Global Variables for Model & Resources ---
model = None
tokenizer = None    
ID2LABEL = None
translator = Translator()


# --- Data Structure for API Request ---
class MessageRequest(BaseModel):
    message: str


# Model loading is handled by the FastAPI lifespan handler `lifespan` defined above.


# --- Helper Function (Prediction Logic) ---
def detect_language(text):
    if re.search(r"[а-яА-ЯёЁ]", text):
        return "ru"
    return "en"

def get_prediction(text: str):
    """Core prediction function, including translation."""
    
    lang = detect_language(text)
    if lang == "ru":
        try:
            translated = translator.translate(text, src="ru", dest="en").text
            text = translated
        except Exception:
            pass # Use original text if translation fails

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LENGTH)
    
    with torch.no_grad():
        outputs = model(**inputs)
        
        logits = outputs.logits
        pred_idx = torch.argmax(logits, dim=1).item()
        probs = torch.nn.functional.softmax(logits, dim=1)[0]
        confidence = torch.max(probs).item() * 100
        
    label = ID2LABEL[pred_idx]
    
    return {
        "text": text,
        "prediction": label,
        "confidence": round(confidence, 2),
    }

# --- API Endpoint ---
@app.post("/predict")
def predict_message(request: MessageRequest):
    """API endpoint to get the spam/ham/scam/toxic classification for a message."""
    try:
        result = get_prediction(request.message)
        return result
    except Exception as e:
        return {"error": str(e), "message": "Prediction failed"}

@app.get("/")
def serve_frontend():
    return FileResponse("index.html")


# --- Health Check Endpoint ---
@app.get("/health")
def health_check():
    """Checks if the server and model are running."""
    return {"status": "ok", "model_loaded": model is not None}