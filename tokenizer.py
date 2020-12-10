import collections
import itertools
import numpy as np

class Tokenizer:
    """字词转ID
    """

    def __init__(self, mintf=16, cutword=False):
        self.char2id = {}
        self.MASK = 0
        self.UNKNOW = 1
        self.mintf = mintf # 最小频率
        self.cutword = cutword
        self.filters = set("!\"#$%&'()[]*+,-./，。！@·……（）【】<>《》?？；‘’“”")
        if cutword:
            import jieba
            jieba.initialize()
            self.lcut = jieba.lcut

    def fit(self, X):
        chars = self.tokenize_and_count(X)
        self.build_vocab(chars)

    def tokenize_and_count(self, X):
        # 统计字
        chars = collections.defaultdict(int)
        for c in itertools.chain(*X):
            chars[c] += 1

        # 统计词
        if self.cutword:
            for x in X:
                for w in self.lcut(x):
                    # 过滤单字的词，因为已经在字表上统计过
                    if len(w) == 1:
                        continue
                    chars[w] += 1
        return chars

    def build_vocab(self, chars):
        # 过滤低频词和特殊符号
        chars = {i: j for i, j in chars.items() \
                 if j >= self.mintf and i not in self.filters}
        # 0:MASK
        # 1:UNK
        # 建立字词ID映射表
        for i, c in enumerate(chars, start=2):
            self.char2id[c] = i

    def fit_generator(self, gen):
        self.fit([gen])

    def fit_in_parallel(self, gen):
        # 并行地构建词表，词频统计上和fit有少许差别
        # gen: generator or iterator
        if self.cutword:
            import jieba
            jieba.initialize()
            # hybrid tokenize
            tokenize = lambda x: itertools.chain((i for i in x), jieba.cut(x))
        else:
            tokenize = lambda x: list(x)

        # from https://github.com/allenwind/count-in-parallel
        from parallel import count_in_parallel_from_generator
        chars = count_in_parallel_from_generator(
            tokenize=tokenize,
            generator=gen,
            processes=6
        )
        self.build_vocab(chars)

    def transform(self, X):
        # 转成ID序列
        ids = []
        for sentence in X:
            s = []
            for char in sentence:
                s.append(self.char2id.get(char, self.UNKNOW))
            s = np.array(s, dtype=np.int32)
            ids.append(s)

        if not self.cutword:
            return ids

        wids = [] # 词ID序列
        wsgs = [] # 词切分ID序列
        for sentence in X:
            wid = []
            wl = []
            for word in self.lcut(sentence):
                w = self.char2id.get(word, self.UNKNOW)
                wid.append(w)
                wl.append(len(word))
            # 字词ID对齐
            wid = np.array(wid, dtype=np.int32)
            w = np.repeat(wid, wl)
            wids.append(w)

            # segment ID对齐
            wsg = np.repeat(np.arange(1, len(wid)+1, dtype=np.int32), wl)
            wsgs.append(wsg)
        return ids, wids, wsgs

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def __len__(self):
        return self.vocab_size

    @property
    def vocab_size(self):
        return len(self.char2id) + 2

    @property
    def vocab(self):
        return self.char2id

    def save(self, file):
        with open(file, "w") as fp:
            json.dump(
                self.char2id,
                fp,
                indent=4,
                ensure_ascii=False
            )

    def load(self, file):
        with open(file, "r") as fp:
            self.char2id = json.load(fp)

def find_best_maxlen(X, mode="mean"):
    # 获取适合的截断长度
    ls = [len(sample) for sample in X]
    if mode == "mode":
        maxlen = np.argmax(np.bincount(ls))
    if mode == "mean":
        maxlen = np.mean(ls)
    if mode == "median":
        maxlen = np.median(ls)
    if mode == "max":
        maxlen = np.max(ls)
    return int(maxlen)

def find_embedding_dims(vocab_size):
    return np.ceil(8.5 * np.log(vocab_size)).astype("int")

def find_max_in_sequences(seqs, offset=1):
    # offset for mask
    return max(np.max(i) for i in seqs) + offset

if __name__ == "__main__":
    # for testing
    texts = ["我爱北京天安门",
             "NLP的魅力在于不断探索",
             "NLP的梦魇在于不断调参",
             "The quick brown fox jumps over the lazy dog"]
    tokenizer = Tokenizer(0, True)
    tokenizer.fit(texts)
    for t, i, j, k in zip(texts, *tokenizer.transform(texts)):
        print(t)
        print(i)
        print(j)
        print(k)
