import tensorflow as tf
from tensorflow.keras.layers import *

def gelu_tanh(x):
    """tanh近似的gelu
    https://arxiv.org/pdf/1606.08415.pdf
    tf.sqrt(2/tf.constant(np.pi)) = 0.7978845608028654
    """
    const = tf.sqrt(2 / tf.constant(np.pi))
    cdf = 0.5 * (1 + tf.tanh(const * (x + 0.044715 * tf.pow(x, 3))))
    return x * cdf

def gelu(x):
    return 0.5 * x * (1.0 + tf.math.erf(x / tf.sqrt(2.0)))

class PositionEmbedding(tf.keras.layers.Layer):
    """可学习的位置Embedding，一种更简单的实现"""

    def __init__(self, maxlen, output_dim, **kwargs):
        super(PositionEmbedding, self).__init__(**kwargs)
        self.maxlen = maxlen
        self.output_dim = output_dim
        self.embedding = tf.keras.layers.Embedding(
            input_dim=maxlen,
            output_dim=output_dim
        )

    def call(self, inputs):
        # maxlen = tf.shape(inputs)[1]
        positions = tf.range(start=0, limit=self.maxlen, delta=1)
        positions = tf.expand_dims(positions, axis=0)
        return self.embedding(positions)

    def compute_output_shape(self, input_shape):
        return input_shape + (self.output_dim,)

    def plot(self):
        import matplotlib.pyplot as plt
        pe = tf.convert_to_tensor(self.embedding.embeddings)
        plt.imshow(pe)
        plt.show()

class Linear(tf.keras.layers.Layer):
    """线性变换"""

    def __init__(self, units, **kwargs):
        super(Linear, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='random_normal',
            rainable=True
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer='random_normal',
            trainable=True
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

class AttentionPooling1D(tf.keras.layers.Layer):

    def __init__(self, hdims, kernel_initializer="glorot_uniform", **kwargs):
        super(AttentionPooling1D, self).__init__(**kwargs)
        self.hdims = hdims
        self.kernel_initializer = tf.keras.initializers.get(
            kernel_initializer
        )
        # time steps dim change
        self.supports_masking = False

    def build(self, input_shape):
        self.k_dense = tf.keras.layers.Dense(
            units=self.hdims,
            kernel_initializer=self.kernel_initializer,
            # kernel_regularizer="l2",
            activation="tanh",
            use_bias=False,
        )
        self.o_dense = tf.keras.layers.Dense(
            units=1,
            # kernel_regularizer="l1", # 添加稀疏性
            use_bias=False
        )

    def call(self, inputs, mask=None):
        if mask is None:
            mask = 1
        else:
            # 扩展维度便于广播
            mask = tf.expand_dims(tf.cast(mask, tf.float32), -1)
        x0 = inputs
        # 计算每个 time steps 权重
        w = self.k_dense(inputs)
        w = self.o_dense(w)
        # 处理 mask
        w = w - (1 - mask) * 1e12
        # 权重归一化
        w = tf.math.softmax(w, axis=1)  # 有mask位置对应的权重变为很小的值
        # 加权平均
        x = tf.reduce_sum(w * x0, axis=1)
        return x

class MaskGlobalMaxPooling1D(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(MaskGlobalMaxPooling1D, self).__init__(**kwargs)

    def call(self, inputs, mask=None):
        if mask is None:
            mask = 1
        else:
            # 扩展维度便于广播
            mask = tf.expand_dims(tf.cast(mask, "float32"), -1)
        x = inputs
        x = x - (1 - mask) * 1e12  # 用一个大的负数mask
        return tf.reduce_max(x, axis=1)
