import os
os.environ["WANDB_DISABLED"] = "true"

from datasets import load_dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from googletrans import Translator
import torch
import re
import json


dataset = load_dataset("alissonpadua/ham-spam-scam-toxic", split="train")


dataset = dataset.select(range(4000))  # можно поставить 2000–5000


unique_labels = list(set(dataset["label"]))
label2id = {label: idx for idx, label in enumerate(unique_labels)}
id2label = {idx: label for label, idx in label2id.items()}
dataset = dataset.map(lambda example: {"label": label2id[example["label"]]})
print("Метки:", label2id)

with open("./eng_spam_model/label_mappings.json", "w", encoding="utf-8") as f:
    json.dump({"id2label": id2label, "label2id": label2id}, f, ensure_ascii=False)

dataset = dataset.train_test_split(test_size=0.2)



tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=64)  # короче тексты

dataset = dataset.map(tokenize, batched=True)
dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])


model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(unique_labels),
    id2label=id2label,
    label2id=label2id
)


training_args = TrainingArguments(
    output_dir="./eng_spam_model",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,   # меньше батч
    per_device_eval_batch_size=4,
    num_train_epochs=1,              # одна эпоха
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
)
trainer.train()


translator = Translator()

def detect_language(text):
    if re.search(r"[а-яА-ЯёЁ]", text):
        return "ru"
    return "en"


def predict(text):
    lang = detect_language(text)
    if lang == "ru":
        translated = translator.translate(text, src="ru", dest="en").text
        print(f"Перевод: {translated}")
        text = translated

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
        conf = torch.max(probs).item() * 100
    label = id2label[pred]
    print(f"Результат: {label} ({conf:.2f}% уверенности)\n")


print("\nПроверка на примерах:\n")
predict("You won a free iPhone! Click here to claim your prize.")
predict("Переведи деньги на этот номер, иначе твой счёт будет заблокирован.")
predict("Привет, как дела?")


print("Теперь можешь вводить свои сообщения. Напиши 'exit' для выхода.\n")
while True:
    msg = input("Введите сообщение: ")
    if msg.lower().strip() == "exit":
        break
    predict(msg)
