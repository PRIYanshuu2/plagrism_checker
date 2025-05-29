import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def prepare_data(df):
    df['sentence1'] = df['sentence1'].apply(clean_text)
    df['sentence2'] = df['sentence2'].apply(clean_text)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df['sentence1'].tolist() + df['sentence2'].tolist())
    seq1 = tokenizer.texts_to_sequences(df['sentence1'])
    seq2 = tokenizer.texts_to_sequences(df['sentence2'])
    max_len = max(max(len(s) for s in seq1), max(len(s) for s in seq2))
    X1 = pad_sequences(seq1, maxlen=max_len, padding='post')
    X2 = pad_sequences(seq2, maxlen=max_len, padding='post')
    return X1, X2, df['is_plagiarized'].values, tokenizer, max_len
