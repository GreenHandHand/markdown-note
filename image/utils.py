# 文本数据预处理
import numpy as np
import torch
import collections
import re
import random


def tokenize(sentences, token="word"):
    """将句子分割为词元"""
    if token == "word":
        return [sentence.split() for sentence in sentences]
    elif token == "char":
        return [list(sentence) for sentence in sentences]
    else:
        print("ERROR: unknown token type " + token)


# 词表
class Vocab:
    """文本词表"""

    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        """
        输入处理后的词元tokens列表, 由tokens初始化一个文本词表
        :param tokens: 词元列表
        :param min_freq: 低于这个概率将被替换为未知词
        :param reserved_tokens: 预留词元，如开始、结束词元等
        """
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 按出现的频率排序
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)

        # 未知词元的索引为0
        self.idx_to_token = ["<unk>"] + reserved_tokens
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}

        for token, freq in self.token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        """词表的单词映射总数"""
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        """索引方法"""
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.token_to_idx["<unk>"])
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indics):
        """将数字索引序列转换为词元"""
        if not isinstance(indics, (list, tuple)):
            return self.idx_to_token[indics]
        return [self.idx_to_token[index] for index in indics]

    @property
    def unk(self):
        """未知词"""
        return 0

    @property
    def token_freqs(self):
        """词元频率"""
        return self._token_freqs


def count_corpus(tokens):
    """统计词元出现的频率"""
    if len(tokens) == 0 or isinstance(tokens[0], list):
        """将词元展平成一个列表"""
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)


class TextDataset:
    """文本处理对象"""

    def __init__(
        self, path, min_freq=0, reserved_tokens=None, token="word", encoding="utf-8"
    ):
        """
        初始化一个 TextLoader 对象。

        :param path: 需要处理的文本的文件路径。
        :param min_freq: 单词的最小出现频率，出现频率低于这个值的单词将被替换为未知词"<unk>"。
        :param reserved_tokens: 预留的单词列表，将这些单词预先添加到词表中以便模型具有更高的优先级。默认为 None。
        :param token: 基本元素是词还是字符，可选项是"word"或"char"。默认为 "word"。
        :param encoding: 文件的编码方式。默认为 "utf-8"。
        """
        self.text_loc = path
        self.token_type = token
        self.encoding = encoding
        self.min_freq = min_freq
        self.reserved_tokens = reserved_tokens

        self.vocab = None
        self.corpus = None

    def build(self, max_tokens=-1):
        """
        处理文本，并生成词表和语料库。

        :param max_tokens: 仅保留前 max_tokens 个最常见的词。
        :return: (list of str) 处理后的语料库， (Vocab) 生成的词表对象。
        """
        with open(self.text_loc, encoding=self.encoding) as f:
            time_machine = f.read()

        lines = [
            re.sub("[^A-Za-z]+", " ", line).strip().lower()
            for line in time_machine.split("\n")
        ]
        print(f"# load sentences : {len(lines)}")
        print(f"Start tokenize the file {self.text_loc} ...")
        tokens = tokenize(lines, self.token_type)
        # 展平成一个列表
        print("successfully tokenized")

        self.vocab = Vocab(tokens, self.min_freq, self.reserved_tokens)
        self.corpus = [self.vocab[token] for line in tokens for token in line]
        print(f"total tokens : {len(self.corpus)}, vocab length : {len(self.vocab)}")
        return self.corpus, self.vocab


def seq_data_iter_random(corpus, batch_size, num_step):
    """使用随机抽样生成一个小批量序列"""
    # 从随机偏移量开始对序列进行分区
    corpus = corpus[random.randint(0, num_step - 1) :]
    # 减去1, 最后一个词元需要作为前一个的标签
    num_sub_seq = (len(corpus) - 1) // num_step
    # 长度为num_sub_seq的子序列的起始索引
    initial_indices = list(range(0, num_sub_seq * num_step, num_step))
    # 随机抽样, 打乱起始索引
    random.shuffle(initial_indices)

    def data(pos):
        """从pos位置开始的长度为num_step的序列"""
        return corpus[pos : pos + num_step]

    num_batches = num_sub_seq // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        # 这里的initial_indices是随机索引
        initial_indices_pre_batch = initial_indices[i : i + batch_size]
        x = [data(j) for j in initial_indices_pre_batch]
        y = [data(j + 1) for j in initial_indices_pre_batch]
        yield torch.tensor(x), torch.tensor(y)


def seq_data_iter_sequential(corpus, batch_size, num_step):
    """使用顺序分区生成一个小批量子序列"""
    # 从随机偏移量开始拆分序列
    offset = random.randint(0, num_step)
    num_tokens = ((len(corpus) - 1 - offset) // batch_size) * batch_size
    Xs = torch.tensor(corpus[offset : offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1 : offset + num_tokens + 1])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_step
    for i in range(0, num_step * num_batches, num_step):
        X = Xs[:, i : i + num_step]
        Y = Ys[:, i : i + num_step]
        yield X, Y


class SeqDataLoader:
    """加载序列数据的迭代器"""

    def __init__(
        self,
        dataset: TextDataset,
        batch_size,
        num_step,
        use_random_iter=True,
        max_tokens=-1,
    ):
        if use_random_iter:
            self.data_iter_fun = seq_data_iter_random
        else:
            self.data_iter_fun = seq_data_iter_sequential
        self.corpus, self.vocab = dataset.build(max_tokens)
        self.batch_size, self.num_step = batch_size, num_step

    def __iter__(self):
        return self.data_iter_fun(self.corpus, self.batch_size, self.num_step)
