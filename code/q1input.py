import pandas as pd
import numpy as np
import nltk
from tqdm import tqdm
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import PunktSentenceTokenizer, RegexpTokenizer

# ==== Input for Q1 with Emotional Variance Feature ====

# ==== INITIAL SETUP ====
tqdm.pandas()
nltk.download('punkt')
nltk.download('vader_lexicon')
tokenizer = PunktSentenceTokenizer()
word_tokenizer = RegexpTokenizer(r'\w+')
vader = SentimentIntensityAnalyzer()

# ==== LOAD & PREPARE DATA ====
df = pd.read_csv("dataset_for_transformers.csv")
df = df[df['post_text'].notnull()]  # get rid of NaN


# ==== Sentence Tokenization ====
def safe_tokenize_sentences(text):
    if pd.isnull(text):
        return []
    return tokenizer.tokenize(str(text))


df['sentences'] = df['post_text'].progress_apply(safe_tokenize_sentences)


# ==== VADER Sentiment per sentence ====
def get_sentiment_stats(sentences):
    scores = [vader.polarity_scores(s)['compound'] for s in sentences]
    if not scores:
        return pd.Series([0, 0, 0])
    return pd.Series([np.mean(scores), np.std(scores), max(scores) - min(scores)])


df[['vader_mean', 'vader_std', 'vader_range']] = df['sentences'].progress_apply(get_sentiment_stats)


# ==== Load NRC Lexicon (with 8 emotions & 2 categories) ====
def load_nrc_lexicon(filepath="NRC-Emolex.csv"):
    lex_df = pd.read_csv(filepath)
    lex_df.columns = lex_df.columns.str.lower()
    emotions = ['anger', 'anticipation', 'disgust', 'fear', 'joy',
                'negative', 'positive', 'sadness', 'surprise', 'trust']
    lexicon = {}

    for _, row in lex_df.iterrows():
        word = row.get('english word', '')
        if pd.isnull(word) or not isinstance(word, str): # skip NaN or float
            continue
        word = word.lower()
        emo_list = [e for e in emotions if row.get(e, 0) == 1]
        if emo_list:
            lexicon[word] = emo_list
    return lexicon, emotions


nrc_dict, emotions = load_nrc_lexicon("NRC-Emolex.csv")


# ==== Count Emotion Per Post ====
def get_emotion_counts(text, lexicon):
    if pd.isnull(text):
        return pd.Series([0] * len(emotions) + [0, 0])

    word_list = word_tokenizer.tokenize(str(text))
    emo_counts = dict.fromkeys(emotions, 0)

    for word in word_list:
        for emo in lexicon.get(word.lower(), []):
            emo_counts[emo] += 1

    values = list(emo_counts.values())
    if not values or sum(values) == 0:
        return pd.Series([0] * len(emotions) + [0, 0])

    return pd.Series(values + [np.std(values), max(values) - min(values)])


emotion_columns = emotions + ['emotion_std', 'emotion_range']
df[emotion_columns] = df['post_text'].progress_apply(lambda x: get_emotion_counts(x, nrc_dict))

# ==== EXPORT FINAL FEATURE SET FOR Q1 ====
final_features = df[['vader_mean', 'vader_std', 'vader_range', 'emotion_std', 'emotion_range'] + emotions + ['label']]
final_features.to_csv("q1_input.csv", index=False)

