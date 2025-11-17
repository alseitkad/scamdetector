import os
import json
from datasets import load_dataset
from transformers import (
    DistilBertTokenizerFast, 
    DistilBertForSequenceClassification, 
    Trainer, 
    TrainingArguments
)

# --- Configuration ---
os.environ["WANDB_DISABLED"] = "true"
MODEL_OUTPUT_DIR = "./eng_spam_model"
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 64
SAMPLE_SIZE = 4000
NUM_TRAIN_EPOCHS = 1
LEARNING_RATE = 2e-5

# Create the output directory if it doesn't exist
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)


def train_and_save_model():
    """Performs data loading, model training, and saves artifacts."""

    print("--- 1. Data Loading and Preparation ---")
    dataset = load_dataset("alissonpadua/ham-spam-scam-toxic", split="train")
    dataset = dataset.select(range(SAMPLE_SIZE))

    # --- Label Mapping ---
    unique_labels = list(set(dataset["label"]))
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    dataset = dataset.map(lambda example: {"label": label2id[example["label"]]})
    print("Labels Mapped:", label2id)

    # Save the label mappings
    with open(f"{MODEL_OUTPUT_DIR}/label_mappings.json", "w", encoding="utf-8") as f:
        json.dump({"id2label": id2label, "label2id": label2id}, f, ensure_ascii=False)

    # Split the dataset
    dataset = dataset.train_test_split(test_size=0.2)
    
    # --- Tokenization ---
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    
    def tokenize(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=MAX_LENGTH)

    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    # --- 2. Model Initialization ---
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(unique_labels),
        id2label=id2label,
        label2id=label2id
    )

    # --- 3. Training ---
    print("\n--- Starting Model Training ---")
    training_args = TrainingArguments(
        output_dir=MODEL_OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=50,
        # Ensure the trained model is saved after the single epoch
        load_best_model_at_end=False, 
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
    )
    
    trainer.train()
    
    # Explicitly save the final model and tokenizer to the output directory
    trainer.save_model(MODEL_OUTPUT_DIR)
    tokenizer.save_pretrained(MODEL_OUTPUT_DIR)
    
    print(f"\n✅ Training complete! Model and tokenizer saved to: {MODEL_OUTPUT_DIR}")


if __name__ == "__main__":
    train_and_save_model()