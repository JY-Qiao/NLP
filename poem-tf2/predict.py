import tensorflow as tf
import numpy as np

from model import lstm_model
from poem import preprocessor, Tokenizer, data_generator

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# 加载语料，并使用分词器转换成向量
file_path = 'dataset/poems.txt'
tokens, poems = preprocessor(file_path)
tokenizer = Tokenizer(tokens)

# 加载模型
model = lstm_model(tokenizer.dict_size)
model.load_weights('logs/poem_writing_weights.h5')

def predict(model, token_idxs):
    total_probs = model.predict([token_idxs, ])[0, -1, 3:]
    idxs = total_probs.argsort()[-100:][::-1]
    probs = total_probs[idxs]
    probs /= sum(probs)
    target_idx = np.random.choice(len(probs), p=probs)

    return idxs[target_idx] + 3

def generate_random_poem(tokenizer, model, text=''):
    MAX_LEN = 80
    token_idxs = tokenizer.encode(text)[:-1]
    while len(token_idxs) < MAX_LEN:
        target = predict(model, token_idxs)
        token_idxs.append(target)
        if target == tokenizer.end_idx:
            break

    return ''.join(tokenizer.decode(token_idxs))

print(generate_random_poem(tokenizer, model, text='江碧鸟逾白'))