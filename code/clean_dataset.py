import pandas as pd
import re


# Load raw data
df = pd.read_csv("Suicide_Detection.csv")

# Drop unnamed index column
df = df.drop(columns=[df.columns[0]])

# Rename columns
df = df.rename(columns={'text': 'post_text', 'class': 'label'})

# Map label: suicide → 1, non-suicide → 0
df['label'] = df['label'].map({'suicide': 1, 'non-suicide': 0})

# Function to remove emoji (Unicode ranges)
def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

# Step 1: Clean for Transformers and Q1 (lowercase + emoji removal)
def basic_clean(text):
    if pd.isnull(text): return ""
    text = text.lower()
    text = remove_emoji(text)
    text = re.sub(r'\s+', ' ', text).strip() # Convert multiple blank spaces to one
    return text

df['post_text'] = df['post_text'].apply(basic_clean)

# Export version for Transformers and Q1 (preserve punctuation)
df[['post_text', 'label']].to_csv("dataset_for_transformers.csv", index=False)

# Step 2: Further clean for Traditional ML (remove punctuation)
def remove_punctuation(text):
    return re.sub(r'[^a-zA-Z0-9\s]', '', text)

df['post_text'] = df['post_text'].apply(remove_punctuation)

# Export version for Traditional ML (no punctuation)
df[['post_text', 'label']].to_csv("dataset.csv", index=False)
