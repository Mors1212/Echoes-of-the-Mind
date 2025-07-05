import time
import tracemalloc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from scipy.sparse import load_npz

# === Load data ===
print("Loading hybrid features...")
X = load_npz("hybrid_features.npz")
y = np.load("labels.npy")

# === Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# === Models to train ===
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "SGDClassifier": SGDClassifier(max_iter=1000, tol=1e-3)
}

results = []

# ====== Confusion Matrix =======
def plot_confusion_matrix(y_true, y_pred, model_name, labels=["Not", "Suicidal"]):
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    filename = f"q5_confmat_{model_name}.png"
    plt.savefig(filename)
    plt.close()
    print(f"[✓] Saved: {filename}")

#===== Learning Curve ====
def plot_learning_curve(model, X, y, model_name):
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=3, scoring='f1', train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=-1
    )
    train_scores_mean = train_scores.mean(axis=1)
    val_scores_mean = val_scores.mean(axis=1)

    plt.figure()
    plt.plot(train_sizes, train_scores_mean, 'o-', label="Training F1")
    plt.plot(train_sizes, val_scores_mean, 'o-', label="Validation F1")
    plt.title(f"Learning Curve - {model_name}")
    plt.xlabel("Training Set Size")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"q5_learning_curve_{model_name}.png")
    plt.close()
    print(f"[✓] Saved: q5_learning_curve_{model_name}.png")


# ======= Training models =========
for name, model in tqdm(models.items(), desc="Training Models"):
    print(f"\nTraining {name}...")
    tracemalloc.start()
    start_time = time.time()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
    print(f"Exec Time: {end_time - start_time:.2f}s | Memory: {peak / 1e6:.2f} MB")

    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "ExecTime_s": end_time - start_time,
        "Memory_MB": peak / 1e6
    })

    #Plot diagrams
    plot_confusion_matrix(y_test, y_pred, name)
    plot_learning_curve(model, X, y, name)

# === Save results ===
results_df = pd.DataFrame(results)
results_df.to_csv("q5_lightweight_results.csv", index=False)
print("\n[✓] Saved: q5_lightweight_results.csv")
print("\n:::Model Comparison Results:")
print(results_df)
print("\n[✓] Finally!!!!!!!")
