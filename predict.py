import os
import json
import re
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from googletrans import Translator

# --- Configuration ---
MODEL_PATH = "./eng_spam_model"
MAX_LENGTH = 64

# --- Load Resources ---
try:
    # 1. Load label mappings
    with open(f"{MODEL_PATH}/label_mappings.json", "r", encoding="utf-8") as f:
        mappings = json.load(f)
    # Keys in JSON are strings, convert to int for ID2LABEL
    ID2LABEL = {int(k): v for k, v in mappings["id2label"].items()} 
    LABEL2ID = mappings["label2id"]
    NUM_LABELS = len(ID2LABEL)
    
    # 2. Load tokenizer and model from the saved path
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_PATH,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )
    model.eval() # Set model to evaluation mode
    translator = Translator()
    print("✅ Model, Tokenizer, and Mappings loaded successfully.")

except Exception as e:
    print(f"❌ Error loading model artifacts from {MODEL_PATH}. Did you run train_model.py?")
    print(f"Details: {e}")
    exit()

# --- Prediction Logic ---

def detect_language(text):
    if re.search(r"[а-яА-ЯёЁ]", text):
        return "ru"
    return "en"

def predict(text, model, tokenizer, id2label):
    """Predicts the classification label for a given text."""
    
    lang = detect_language(text)
    if lang == "ru":
        try:
            translated = translator.translate(text, src="ru", dest="en").text
            print(f"Перевод: {translated}")
            text = translated
        except Exception:
            # googletrans can fail sometimes
            print("Translation failed. Using original text.")

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LENGTH)
    
    with torch.no_grad():
        outputs = model(**inputs)
        
        pred = torch.argmax(outputs.logits, dim=1).item()
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
        conf = torch.max(probs).item() * 100
        
    label = id2label[pred]
    return {"prediction": label, "confidence": conf}

# --- Interactive Test Run ---
if __name__ == "__main__":
    print("\n--- Starting Interactive Prediction Test ---")

    test_examples = [
        "You won a free iPhone! Click here to claim your prize.",
        "Переведи деньги на этот номер, иначе твой счёт будет заблокирован.",
        "Привет, как дела?",
    ]
    
    for text in test_examples:
        print(f"\nINPUT: {text}")
        result = predict(text, model, tokenizer, ID2LABEL)
        print(f"Результат: {result['prediction']} ({result['confidence']:.2f}% уверенности)")

    while True:
        msg = input("Введите сообщение: ")
        if msg.lower().strip() == "exit":
            break
        result = predict(msg, model, tokenizer, ID2LABEL)
        print(f"Результат: {result['prediction']} ({result['confidence']:.2f}% уверенности)")