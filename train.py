import pandas as pd
from sklearn.model_selection import train_test_split
from src.preprocessing import prepare_data
from src.model import build_siamese_model

def train_model():
    df = pd.read_csv("data/train.csv")
    X1, X2, y, tokenizer, max_len = prepare_data(df)
    vocab_size = len(tokenizer.word_index) + 1
    X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(X1, X2, y, test_size=0.2, random_state=42)

    model = build_siamese_model(vocab_size, max_len)
    model.fit([X1_train, X2_train], y_train, validation_data=([X1_test, X2_test], y_test), epochs=5, batch_size=64)
    model.save("models/siamese_lstm.h5")
    return model, tokenizer, max_len
