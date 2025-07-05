import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, save_npz

# === Step 1: Load dataset ===
df = pd.read_csv("q3-5_input.csv")
df["post_text"] = df["post_text"].fillna("")

# === Step 2: TF-IDF vectorization ===
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words="english")
X_tfidf = vectorizer.fit_transform(df["post_text"])

# === Step 3: Load emotion features ===
emotion_cols = ['anger', 'anticipation', 'disgust', 'fear', 'joy',
                'negative', 'positive', 'sadness', 'surprise', 'trust',
                'emotion_std', 'emotion_range']

X_emotion = df[emotion_cols].fillna(0).values

# === Step 4: Combine TF-IDF + emotion features ===
X_combined = hstack([X_tfidf, X_emotion])
y = df["label"].values

# === Step 5: Save everything for Q3 training ===
save_npz("hybrid_features.npz", X_combined)
np.save("labels.npy", y)
np.save("tfidf_vocab.npy", vectorizer.get_feature_names_out())

print("[✓] Saved: hybrid_features.npz, labels.npy, tfidf_vocab.npy")
print(f"Feature shape: {X_combined.shape}, Labels: {len(y)}")
