import tensorflow as tf
from tensorflow.keras.layers import LSTM, TimeDistributed, Dense

def lstm_model(input_dim):

    model = tf.keras.Sequential()
    # 词嵌入层
    model.add(tf.keras.layers.Embedding(input_dim=input_dim, output_dim=150))
    # 第一个LSTM层
    model.add(LSTM(150, dropout=0.5, return_sequences=True))
    # 第二个LSTM层
    model.add(LSTM(150, dropout=0.5, return_sequences=True))
    # Dense层
    model.add(TimeDistributed(Dense(input_dim, activation='softmax')))

    return model