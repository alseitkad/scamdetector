from datasets import load_dataset

print("Загружаем датасет...")
dataset = load_dataset("alissonpadua/ham-spam-scam-toxic")

train_data = dataset["train"]

output_path = "dataset.csv"
train_data.to_csv(output_path)

print(f" Датасет успешно сохранён: {output_path}")
