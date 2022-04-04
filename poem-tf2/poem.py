import numpy as np
from collections import Counter
import math
import re

UNUSED_WORDS = ['(', ')','（', '）',  '[', ']', '【', '】', '《', '》', '_']
poems = []

def preprocessor(file_path):
    poems = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        fields = re.split(r"[:：]", line)
        if len(fields) != 2:
            continue
        content = fields[1]
        if any(word in content for word in UNUSED_WORDS):
            continue
        if len(content) < 5 or len(content) > 80:
            continue
        poems.append(content.replace('\n', ''))

    # 统计词频
    counter = Counter()
    for poem in poems:
        counter.update(poem)

    # 过滤掉低词频的词
    MIN_WORD_FREQUENCY = 8
    tokens = [token for token, count in counter.items() if count >= MIN_WORD_FREQUENCY]

    # 补上填充字符，None字符，开始字符，结束字符
    tokens = ['[PAD]', '[NONE]', '[START]', '[END]'] + tokens

    return tokens, poems

class Tokenizer:
    # 将每首诗转换成数字向量
    def __init__(self, tokens):
        # 词汇表大小
        self.dict_size = len(tokens)
        # 生成映射关系
        self.token_idx = {}  # 映射字典:词-->数字
        self.idx_token = {}  # 映射字典:数字-->词
        for idx, word in enumerate(tokens):
            self.token_idx[word] = idx
            self.idx_token[idx] = word

        # 赋予特殊标记idx
        self.start_idx = self.token_idx['[START]']
        self.end_idx = self.token_idx['[END]']
        self.none_idx = self.token_idx['[NONE]']
        self.pad_idx = self.token_idx['[PAD]']

    def idx_to_token(self, token_idx):
        return self.idx_token.get(token_idx)

    def token_to_idx(self, token):
        return self.token_idx.get(token, self.none_idx)

    def encode(self, tokens):
        token_idxs = [self.start_idx, ]

        for token in tokens:
            token_idxs.append(self.token_to_idx(token))
        token_idxs.append(self.end_idx)

        return token_idxs

    def decode(self, token_idxs):
        flag_tokens = {'[START]', '[END]'}

        tokens = []
        for idx in token_idxs:
            token = self.idx_to_token(idx)
            if token not in flag_tokens:
                tokens.append(token)

        return tokens

class data_generator:
    def __init__(self, data, tokenizer, batch_size):
        self.data = data
        self.total_size = len(self.data)
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.steps = int(math.floor(len(self.data) / self.batch_size))

    def pad_line(self, line, length, padding=None):
        if padding is None:
            padding = self.tokenizer.pad_idx
        padding_length = length - len(line)
        if padding_length > 0:
            return line + [padding] * padding_length
        else:
            return line[:length]

    def __len__(self):
        return self.steps

    def __iter__(self):
        np.random.shuffle(self.data)

        for start in range(0, self.total_size, self.batch_size):
            end = min(start + self.batch_size, self.total_size)
            data = self.data[start:end]

            max_length = max(map(len, data))
            batch_data = []
            for str_line in data:
                # 对每一行的诗词进行编码并补充padding
                encode_line = self.tokenizer.encode(str_line)
                pad_encode_line = self.pad_line(encode_line, max_length + 2)
                batch_data.append(pad_encode_line)
            batch_data = np.array(batch_data)

            yield batch_data[:, :-1], batch_data[:, 1:]

    def generator(self):
        while True:
            yield from self.__iter__()