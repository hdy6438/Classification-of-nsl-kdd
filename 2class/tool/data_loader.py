import random
from math import ceil
import keras
import numpy as np


class DataGenerator(keras.utils.Sequence):

    def __init__(self, path,net_name, batch_size=128):
        dataset = np.load(path)  # 加载数据
        self.batch_size = batch_size
        self.net_name = net_name
        self.data_x = dataset['x']
        self.data_y = dataset['y']
        self.indexs = list(range(len(self.data_x)))
        print(self.data_x.shape)
        print(self.data_y.shape)

    def __len__(self):
        return int(ceil(len(self.data_x) / self.batch_size))

    def __getitem__(self, index):
        # print(self.data_y[index])
        if self.net_name == "rnn" or self.net_name == "lstm" :
            return np.expand_dims(self.data_x[index].reshape((-1,1)), 0), np.expand_dims(self.data_y[index], 0)
        elif self.net_name == "dnn":
            return np.expand_dims(self.data_x[index], 0), np.expand_dims(self.data_y[index], 0)
        else:
            raise ValueError("net name error")

    def on_epoch_end(self):
        random.shuffle(self.indexs)
        self.data_x = self.data_x[self.indexs]
        self.data_y = self.data_y[self.indexs]



