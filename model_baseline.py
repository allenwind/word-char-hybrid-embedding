import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices("GPU")
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from sklearn.model_selection import train_test_split
from sklearn import metrics

from dataset import load_THUCNews_title_label
from tokenizer import Tokenizer, find_best_maxlen
from tflayers import AttentionPooling1D, PositionEmbedding, gelu
from tfutils import SaveBestModelOnMemory

# 纯字Embedding
# 92%+

# 处理数据
X, y, classes = load_THUCNews_title_label()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, random_state=7322
)

num_classes = len(classes)
# 转化成字id
print("tokenize...")
tokenizer = Tokenizer(mintf=32, cutword=False)
tokenizer.fit(X_train)

# maxlen = find_best_maxlen(X_train, mode="max")
maxlen = 48

def create_dataset(X, y, maxlen):
    X = tokenizer.transform(X)
    X = sequence.pad_sequences(
      X,
      maxlen=maxlen,
      dtype="int32",
      padding="post",
      truncating="post",
      value=0.0
    )
    y = tf.keras.utils.to_categorical(y)
    return X, y

X_train, y_train = create_dataset(X_train, y_train, maxlen=maxlen)

# 模型
num_words = len(tokenizer)
embedding_dims = 128

inputs = Input(shape=(maxlen,))
x = inputs
mask = Lambda(lambda x: tf.not_equal(x, 0))(x)
embedding = Embedding(
    num_words,
    embedding_dims,
    embeddings_initializer="uniform",
    input_length=maxlen
)
pembedding = PositionEmbedding(
    maxlen=maxlen,
    output_dim=embedding_dims
)

x = embedding(x)
p = pembedding(x)
x = p + x
x = LayerNormalization()(x)
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

model = Model(inputs, outputs)
model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])
model.summary()

# 训练
batch_size = 32
epochs = 5
callbacks = [SaveBestModelOnMemory()]
model.fit(
    X_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=callbacks,
    validation_split=0.1
)

# 评估
X_test, y_test = create_dataset(X_test, y_test, maxlen=maxlen)
model.evaluate(X_test, y_test)
