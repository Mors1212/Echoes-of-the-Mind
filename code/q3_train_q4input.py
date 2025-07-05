import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from scipy.sparse import load_npz
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
import shap

# === Load data ===
print("Loading features and labels...")
X = load_npz("hybrid_features.npz")
y = np.load("labels.npy")

# === Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# === Define models ===
models = {
    "LogisticRegression": LogisticRegression(max_iter=3000),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "NaiveBayes": MultinomialNB()
}

results = []

# === Confusion matrix Function ===
def plot_confusion_matrix(y_true, y_pred, model_name, prefix="q3"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Non-suicidal", "Suicidal"],
                yticklabels=["Non-suicidal", "Suicidal"])
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    filename = f"{prefix}_confusion_{model_name.lower()}.png"
    plt.savefig(filename)
    plt.close()
    print(f"\n[✓] Saved: {filename}")

# === Learning Curve Function ===
def plot_learning_curve(model, X, y, name):
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=5, scoring='f1', n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 5)
    )
    train_mean = np.mean(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)

    plt.figure()
    plt.plot(train_sizes, train_mean, 'o-', label="Training F1")
    plt.plot(train_sizes, val_mean, 'o-', label="Validation F1")
    plt.title(f"Learning Curve - {name}")
    plt.xlabel("Training Set Size")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    filename = f"q3_learning_curve_{name.lower()}.png"
    plt.savefig(filename)
    plt.close()
    print(f"\n[✓] Saved: {filename}")

# === Train models with detailed logs ===
print("\nTraining models...")

for name, model in tqdm(models.items(), desc="Training"):
    print(f"\n Training: {name}")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # Save results
    results.append({
        "model": name,
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds)
    })

    # Plot diagrams
    plot_confusion_matrix(y_test, preds, name, prefix="q3")
    plot_learning_curve(model, X_train, y_train, name)

# === Save comparison results ===
results_df = pd.DataFrame(results)
results_df.to_csv("q3_model_results.csv", index=False)

print("\n:::Model Comparison Results:")
print(results_df.to_string(index=False))

# === SHAP analysis for Logistic Regression for Q4 ===
print("\n SHAP (LogisticRegression) analyzing feature impact...")
explainer = shap.Explainer(models["LogisticRegression"], X_train[:200])
shap_values = explainer(X_test[:200])
print("Saving SHAP values and TF-IDF vocab for Word Cloud...")
np.save("lr_shap_values_logreg.npy", shap_values.values)
print("\n[✓] Saved: lr_shap_values_logreg.npy")
print("\n[✓] Done!")
