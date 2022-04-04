import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import sparse_categorical_crossentropy

from model import lstm_model
from poem import preprocessor, Tokenizer, data_generator

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# 定义训练参数
batch_size = 64
learning_rate = 0.001
file_path = 'dataset/poems.txt'
epochs = 100

tokens, poems = preprocessor(file_path)
tokenizer = Tokenizer(tokens)
dataset = data_generator(poems, tokenizer, batch_size)

# 建立网络
model = lstm_model(tokenizer.dict_size)
# 编译网络
model.compile(optimizer=Adam(learning_rate), loss=sparse_categorical_crossentropy)
# 训练网络
model.fit(dataset.generator(), steps_per_epoch=dataset.steps, epochs=epochs)
# 保存结果
model.save_weights('logs/poem_writing_weights.h5')