import pandas as pd
from nltk.tokenize import RegexpTokenizer

# === Step 1: Load dataset ===
df = pd.read_csv("dataset.csv")
df["post_text"] = df["post_text"].fillna("")

# === Step 2: Load NRC Lexicon ===
nrc_df = pd.read_csv("NRC-Emolex.csv")
nrc_lexicon = nrc_df.set_index("English Word")
nrc_emotions = nrc_lexicon.columns.tolist()

# === Step 3: Prepare tokenizer ===
tokenizer = RegexpTokenizer(r'\w+')

# === Step 4: Extract emotion features ===
def extract_emotions(text):
    words = tokenizer.tokenize(text.lower())
    counts = dict.fromkeys(nrc_emotions, 0)
    for word in words:
        if word in nrc_lexicon.index:
            for emo in nrc_emotions:
                counts[emo] += nrc_lexicon.loc[word, emo]
    return pd.Series(counts)

# Apply to entire dataset
emotion_scores = df["post_text"].apply(extract_emotions)

# === Step 5: Compute emotion_std and emotion_range ===
df["emotion_std"] = emotion_scores.std(axis=1)
df["emotion_range"] = emotion_scores.max(axis=1) - emotion_scores.min(axis=1)

# === Step 6: Merge and Export ===
df = pd.concat([df, emotion_scores], axis=1)
df.to_csv("q3-5_input.csv", index=False)
