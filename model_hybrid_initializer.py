import tensorflow as tf

# physical_devices = tf.config.experimental.list_physical_devices("GPU")
# assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from sklearn.model_selection import train_test_split
from tensorflow.keras import initializers
from sklearn import metrics
import numpy as np

from dataset import load_THUCNews_title_label
from tokenizer import Tokenizer, find_best_maxlen
from tflayers import AttentionPooling1D, PositionEmbedding, gelu
from hybrid import CharAlignHybridEmbedding
from tfutils import SaveBestModelOnMemory

def load_char2vec(file="weights/chinese_L-12_H-768_A-12_Embedding.hdf5"):
    import h5py
    fp = h5py.File(file, "r")
    group = fp["embeddings"]
    tokens = list(group.keys())
    char2vec = {}
    for token in tokens:
        char2vec[token] = np.array(group[token])
    return char2vec

class HybridInitializer(initializers.Initializer):
    """字词混合初始化，词的向量是其对应字的向量的均值，
    UNK使用所有字向量的均值"""

    def __init__(self, output_dim, char2vec, hybrid2id):
        input_dim = len(hybrid2id) + 2
        self.embeddings = np.zeros((input_dim, output_dim))
        vecs = list(char2vec.values())
        self.embeddings[1] = np.mean(vecs, axis=0)
        for hybrid, _id in hybrid2id.items():
            if self._is_word(hybrid):
                vec = 0.0
                for char in hybrid:
                    vec += char2vec.get(char, self.embeddings[1])
                vec = vec / len(hybrid)
                self.embeddings[_id] = vec
            else:
                vec = char2vec.get(hybrid, self.embeddings[1])
                self.embeddings[_id] = vec

    def __call__(self, shape, dtype=None):
        return tf.cast(self.embeddings, dtype)

    def _is_word(self, hybrid):
        return len(hybrid) > 1

# 字词混合Embedding
# 93.5%+

# 处理数据
X, y, classes = load_THUCNews_title_label()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, random_state=7322
)

num_classes = len(classes)
# 转化成字id
print("tokenize...")
tokenizer = Tokenizer(mintf=32, cutword=True)
tokenizer.fit_in_parallel(X_train)

# maxlen = find_best_maxlen(X_train, mode="max")
maxlen = 48

def pad(X, maxlen):
    return sequence.pad_sequences(
        X,
        maxlen=maxlen,
        dtype="int32",
        padding="post",
        truncating="post",
        value=0
        )

def create_dataset(X, y, maxlen):
    Xc, Xw, Xs = tokenizer.transform_in_parallel(X)
    Xc = pad(Xc, maxlen)
    Xw = pad(Xw, maxlen)
    Xs = pad(Xs, maxlen)
    y = tf.keras.utils.to_categorical(y)
    return Xc, Xw, Xs, y

Xc_train, Xw_train, Xs_train, y_train = create_dataset(
    X_train,
    y_train,
    maxlen=maxlen
)

# 模型
char2vec = load_char2vec()
num_words = len(tokenizer)
embedding_dims = 768
initializer = HybridInitializer(embedding_dims, char2vec, tokenizer.vocab)

c_input = Input(shape=(maxlen,)) # 字
w_input = Input(shape=(maxlen,)) # 词
s_input = Input(shape=(maxlen,)) # segment

mask = Lambda(lambda x: tf.not_equal(x, 0))(c_input)
hybridembedding = CharAlignHybridEmbedding(
    input_dim=num_words,
    output_dim=embedding_dims,
    hybridmerge="add",
    max_segment_length=maxlen,
    without_segment_embedding=True,
    embeddings_initializer=initializer
)
pembedding = PositionEmbedding(
    maxlen=maxlen,
    output_dim=embedding_dims
)

x = hybridembedding([c_input, w_input, s_input])
p = pembedding(x)
x = p + x
x = Dropout(0.1)(x)
x = Conv1D(filters=128,
           kernel_size=2,
           padding="same",
           activation=gelu,
           strides=1)(x)
x = Conv1D(filters=128,
           kernel_size=3,
           padding="same",
           activation=gelu,
           strides=1)(x)
x = AttentionPooling1D(hdims=128)(x, mask=mask)
x = Dropout(0.2)(x)
x = Dense(128)(x)
x = gelu(x)
outputs = Dense(num_classes, activation="softmax")(x)

model = Model([c_input, w_input, s_input], outputs)
model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])
model.summary()

# 训练
batch_size = 32
epochs = 30
callbacks = [SaveBestModelOnMemory()]
model.fit(
    [Xc_train, Xw_train, Xs_train],
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=callbacks,
    validation_split=0.1
)

# 评估
Xc_test, Xw_test, Xs_test, y_test = create_dataset(
    X_test,
    y_test,
    maxlen=maxlen
)
model.evaluate([Xc_test, Xw_test, Xs_test], y_test)
