import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback

class SaveBestModelOnMemory(Callback):

    def __init__(self, monitor='val_loss', monitor_op=np.less, period=1, path=None):
        super(SaveBestModelOnMemory, self).__init__()
        self.monitor = monitor
        self.monitor_op = monitor_op

        # 每 period 个 epoch 进行一次 monitor_op,默认一次,这也是比较好理解的
        self.period = period
        # 如果存在，则保存最优模型到指定路径
        self.path = path
        # 记录最优权重的 epoch,用于调参时调整 epoch 以减少训练时间
        self.best_epochs = 0
        self.epochs_since_last_save = 0
        self.best = np.Inf

    def on_train_begin(self, logs=None):
        self.best_weights = self.model.get_weights()

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}

        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0

            current = logs.get(self.monitor, None)
            if current is None:
                print(self.monitor, "not support")
            else:
                # 满足条件更新权重
                if self.monitor_op(current, self.best):
                    # 记录当前的最优指标值
                    self.best = current
                    self.best_epochs = epoch + 1
                    # 更新最优权重
                    self.best_weights = self.model.get_weights()

    def on_train_end(self, logs=None):
        print("best epoch on", self.best_epochs)
        print("best loss", self.best)
        # 保存模型到指定路径
        if self.path is not None:
            self.model.save(self.path)

        # 训练结束时设置最优权重
        self.model.set_weights(self.best_weights)
