import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns

# === LOAD DATA ===
df = pd.read_csv("q1_input.csv")

# === DEFINE FEATURE SETS ===
emotion_cols = ['anger', 'anticipation', 'disgust', 'fear', 'joy',
                'sadness', 'surprise', 'trust']

variance_cols = ['vader_mean', 'vader_std', 'vader_range', 'emotion_std', 'emotion_range']

X_base = df[emotion_cols]
X_var = df[emotion_cols + variance_cols]
y = df['label']

# === SPLIT DATA ===
Xb_train, Xb_test, yb_train, yb_test = train_test_split(X_base, y, test_size=0.2, random_state=42, stratify=y)
Xv_train, Xv_test, yv_train, yv_test = train_test_split(X_var, y, test_size=0.2, random_state=42, stratify=y)

# === EVALUATION FUNCTION ===
def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return {
        "features": name,
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds)
    }, preds

# === CONFUSION MATRIX ===
def plot_confusion(name, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Non-suicidal", "Suicidal"],
                yticklabels=["Non-suicidal", "Suicidal"])
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    save_name = f"q1_confmat_{name.lower()}.png"
    plt.savefig(save_name)
    plt.close()
    print(f"\n[✓] Saved: {save_name}")

# === LEARNING CURVE ===
def plot_learning_curve(estimator, X, y, name="model"):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, scoring='f1', train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=-1
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    plt.figure(figsize=(6, 4))
    plt.plot(train_sizes, train_scores_mean, 'o-', label="Training F1")
    plt.plot(train_sizes, test_scores_mean, 'o-', label="Validation F1")
    plt.title(f"{name} - Learning Curve (F1)")
    plt.xlabel("Training Size")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    save_name = f"q1_learning_curve_{name.lower()}.png"
    plt.savefig(save_name)
    plt.close()
    print(f"\n[✓] Saved: {save_name}")

# === DEFINE TRAINING SETTINGS ===
settings = [
    ("Baseline", Xb_train, Xb_test, yb_train, yb_test, X_base),
    ("Variance-enhanced", Xv_train, Xv_test, yv_train, yv_test, X_var)
]

results = []

# === TRAINING LOOP ===
for name, X_train, X_test, y_train, y_test, X_all in tqdm(settings, desc="Training"):
    print(f"\nTraining ({name})...")
    model = LogisticRegression(max_iter=1000)
    metrics, preds = evaluate_model("LogisticRegression", model, X_train, X_test, y_train, y_test)
    metrics["features"] = name.lower()
    results.append(metrics)

    plot_confusion(name, y_test, preds)
    plot_learning_curve(model, X_all, y, name=name)

# === RESULTS TABLE ===
results_df = pd.DataFrame(results)
print("\n:::Model Comparison Results:")
print(results_df[['features', 'accuracy', 'precision', 'recall', 'f1']].to_string(index=False))

results_df.to_csv("q1_results.csv", index=False)
print("[✓] Done")