import tensorflow as tf
from tensorflow.keras.layers import *

class CharAlignHybridEmbedding(tf.keras.layers.Layer):
    """字词混合Embedding，以字为基准对齐"""

    def __init__(
        self,
        input_dim,
        output_dim,
        hybridmerge="add",
        max_segment_length=100,
        without_segment_embedding=False,
        embeddings_initializer="uniform",
        **kwargs):
        super(CharAlignHybridEmbedding, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hybridmerge = hybridmerge if hybridmerge in ("add", "concat") else "add"
        self.max_segment_length = max_segment_length # 词的最大字长度
        self.without_segment_embedding = without_segment_embedding
        if isinstance(embeddings_initializer, str):
            embeddings_initializer = tf.keras.initializers.get(embeddings_initializer)
        self.embeddings_initializer = embeddings_initializer

    def build(self, input_shape):
        self.embeddings = Embedding(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            embeddings_initializer=self.embeddings_initializer
        )
        if not self.without_segment_embedding:
            self.segment_embeddings = Embedding(
                input_dim=self.max_segment_length,
                output_dim=self.output_dim,
                embeddings_initializer=self.embeddings_initializer
            )
        if self.hybridmerge == "concat":
            self.o_dense = Dense(self.output_dim)

    def call(self, inputs, mask=None):
        # 字ID，词ID，段ID
        cids, wids, sids = inputs
        if mask is None:
            mask = 1.0
        else:
            mask = tf.expand_dims(tf.cast(mask, tf.float32), axis=-1)

        xc = self.embeddings(cids)
        xw = self.embeddings(wids)
        if not self.without_segment_embedding:
            xs = self.segment_embeddings(sids)
        else:
            xs = 0.0

        if self.hybridmerge == "add":
            u = 2.0 if self.without_segment_embedding else 3.0
            # 叠加后要scale会原来区间
            x = xc + xw + xs / u
        else:
            x = tf.concat([xc, xw, xs], axis=-1)
            # 融合三个Embedding信息
            x = self.o_dense(x)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape + (self.output_dim,)
