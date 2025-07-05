from wordcloud import WordCloud
from matplotlib import pyplot as plt
import numpy as np
import json
import pandas as pd

# Load vocab list
with open("tfidf_vocab.json", "r") as f:
    vocab = json.load(f)

# Load SHAP values & vocab
shap_values = np.load("lr_shap_values_logreg.npy", allow_pickle=True)
vocab_size = len(vocab)
tfidf_shap_values = shap_values.mean(axis=0)[:vocab_size]

# Split into positive and negative SHAP words
positive_words = {}
negative_words = {}
for idx, shap_val in enumerate(tfidf_shap_values):
    word = vocab[idx]
    if shap_val > 0:
        positive_words[word] = shap_val
    elif shap_val < 0:
        negative_words[word] = -shap_val

#=====Word Cloud===
def dark_red_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    return (180, 0, 0)  # dark red

def dark_blue_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    return (30, 60, 150)  # dark blue

# Build and recolor Word Clouds
wordcloud_pos = WordCloud(width=1600, height=800, background_color='white').generate_from_frequencies(positive_words)
wordcloud_pos.recolor(color_func=dark_red_color_func)


wordcloud_neg = WordCloud(width=1600, height=800, background_color='white').generate_from_frequencies(negative_words)
wordcloud_neg.recolor(color_func=dark_blue_color_func)

# Save as high-res PNG
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_pos, interpolation='bilinear')
plt.axis('off')
plt.tight_layout()
plt.savefig("q4_logreg_wc_positive.png", dpi=300)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_neg, interpolation='bilinear')
plt.axis('off')
plt.tight_layout()
plt.savefig("q4_logreg_wc_negative.png", dpi=300)

# Save top 40 SHAP words to CSV
top_n = 40
top_pos = sorted(positive_words.items(), key=lambda x: x[1], reverse=True)[:top_n]
top_neg = sorted(negative_words.items(), key=lambda x: x[1], reverse=True)[:top_n]

df_top = pd.DataFrame({
    "word": [w for w, _ in top_pos + top_neg],
    "shap_value": [v for _, v in top_pos + top_neg],
    "direction": ["positive (↑ risk)"] * len(top_pos) + ["negative (↓ risk)"] * len(top_neg)
})

df_top.to_csv("q4_logreg_top40_words.csv",index=False)