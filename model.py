from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Lambda
import tensorflow.keras.backend as K

def build_siamese_model(vocab_size, max_len, embedding_dim=128):
    def base_network(input_shape):
        input = Input(shape=input_shape)
        x = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(input)
        x = LSTM(64)(x)
        return Model(input, x)

    input_shape = (max_len,)
    base = base_network(input_shape)
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)
    processed_a = base(input_a)
    processed_b = base(input_b)
    L1 = Lambda(lambda x: K.abs(x[0] - x[1]))([processed_a, processed_b])
    output = Dense(1, activation='sigmoid')(L1)
    model = Model([input_a, input_b], output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
