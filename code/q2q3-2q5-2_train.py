import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tracemalloc
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    RobertaTokenizer, RobertaForSequenceClassification,
    DistilBertTokenizer, DistilBertForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding
)
from datasets import Dataset

# If run time is too long, you can separate and run each model one by one

# ====Load dataset====
print("Loading dataset...")
df = pd.read_csv("dataset_for_transformers.csv")
df = df[["post_text", "label"]].dropna()
df["label"] = df["label"].astype(int)

# ====Train-test split====
from sklearn.model_selection import train_test_split
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["post_text"].tolist(), df["label"].tolist(), test_size=0.2, random_state=42
)

MODELS = {
    "RoBERTa": {
        "tokenizer": RobertaTokenizer.from_pretrained("roberta-base"),
        "model": RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
    },
    "DistilBERT": {
        "tokenizer": DistilBertTokenizer.from_pretrained("distilbert-base-uncased"),
        "model": DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    },
    "BERT": {
        "tokenizer": BertTokenizer.from_pretrained("bert-base-uncased"),
        "model": BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    }
}

results = []

# ==== Training models ====
for model_name, components in tqdm(list(MODELS.items()), desc="Training Models"):
    print(f"\n===== Training {model_name} =====")
    tokenizer = components["tokenizer"]
    model = components["model"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)

    # Tokenization
    print("Tokenizing...")
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True)

    # Dataset formating
    # make hugging face datasets for train and test
    train_dataset = Dataset.from_dict({
        "input_ids": train_encodings["input_ids"],
        "attention_mask": train_encodings["attention_mask"],
        "labels": train_labels
    })
    test_dataset = Dataset.from_dict({
        "input_ids": test_encodings["input_ids"],
        "attention_mask": test_encodings["attention_mask"],
        "labels": test_labels
    })

    # for batching (optimize memo.)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=f"./results_{model_name}",
        num_train_epochs=2,
        per_device_train_batch_size=20,
        per_device_eval_batch_size=32,
        logging_strategy="steps",
        logging_steps=4000, # for learning curve
        logging_dir=f"./logs_{model_name}",
        report_to="none",
        disable_tqdm=False
    )

    # Trainer API
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    print("Training...")
    torch.cuda.empty_cache()
    # track time, memory, and GPU memory
    tracemalloc.start()
    start_time = time.time()
    gpu_mem_start = torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else None
    gpu_mem_peak_start = torch.cuda.max_memory_reserved() / 1e6 if torch.cuda.is_available() else None

    # training & Evaluation
    trainer.train()
    preds = trainer.predict(test_dataset)
    y_pred = np.argmax(preds.predictions, axis=1)
    y_true = preds.label_ids

    # end of tracking
    gpu_mem_end = torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else None
    gpu_mem_peak_end = torch.cuda.max_memory_reserved() / 1e6 if torch.cuda.is_available() else None
    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Evaluation metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # print result after done each model
    print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
    print(f"Execution Time: {end_time - start_time:.2f}s | Peak Memory Usage: {peak / 1e6:.2f} MB")
    if gpu_mem_start is not None:
        print(f"    - GPU Used: {gpu_mem_end - gpu_mem_start:.2f} MB | Peak: {gpu_mem_peak_end:.2f} MB")

    results.append({
        "Model": model_name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "ExecTime_s": end_time - start_time,
        "Memory_MB": peak / 1e6,
        "GPU_Used_MB": (gpu_mem_end - gpu_mem_start) if gpu_mem_start is not None else None,
        "GPU_Peak_MB": gpu_mem_peak_end if gpu_mem_peak_end is not None else None
    })

    #==== Confusion matrix====
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Not", "Suicidal"], yticklabels=["Not", "Suicidal"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.savefig(f"q2_confmat_{model_name}.png")
    plt.close()
    print(f"[✓] Saved: q2_confmat_{model_name}.png")

    #====Learning curve====
    logs = trainer.state.log_history
    train_logs = [(l['step'], l['loss']) for l in logs if 'loss' in l]

    if train_logs:
        train_steps, train_losses = zip(*train_logs)


        def smooth(vals, k=3):
            if len(vals) < k:
                return vals
            return np.convolve(vals, np.ones(k) / k, mode='valid')


        smooth_k = 3
        smoothed_train_losses = smooth(train_losses, k=smooth_k)
        smoothed_train_steps = train_steps[(smooth_k - 1) // 2: len(smoothed_train_losses) + (smooth_k - 1) // 2]

        plt.figure(figsize=(8, 5))
        plt.plot(smoothed_train_steps, smoothed_train_losses, label="Train Loss (smoothed)")

        plt.title(f"Learning Curve - {model_name}")
        plt.xlabel("Training Steps")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(f"q2_learning_curve_{model_name}.png")
        plt.close()
        print(f"[✓] Saved: q2_learning_curve_{model_name}.png")

# Save final results
results_df = pd.DataFrame(results)
results_df.to_csv("q2_model_comparison.csv", index=False)
print("\n:::Model Comparison Results:")
print(results_df)
print("\n[✓] All done!")

