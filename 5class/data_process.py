import os.path
import random

import numpy as np
import pandas
from imblearn.over_sampling import SMOTE,RandomOverSampler,SVMSMOTE

from tool.ig import IG
from tool.map import *
from setting import path
from keras.utils import np_utils



if __name__ == "__main__":

    print("loading data")
    train_np = pandas.read_csv(os.path.join(path.data_path, "train.csv")).to_numpy()  # 加载训练数据
    test_np = pandas.read_csv(os.path.join(path.data_path, "test.csv")).to_numpy()  # 加载测试数据

    print("data load successfully,processing")

    train_x = train_np[:, :-2]
    test_x = test_np[:, :-2]

    train_y = np.array([label.one_hot_map[label.map_dict[name]] for name in train_np[:, -2]])
    test_y = np.array([label.one_hot_map[label.map_dict[name]] for name in test_np[:, -2]])

    #数值编码
    def numerical_coding(dataset):
        res = np_utils.to_categorical([Protocol_type.one_hot_map[name] for name in dataset[:, 1]], 3)
        dataset = np.delete(dataset, 1, axis=1)
        for i in range(3):
            dataset = np.insert(dataset, 1+i, res[:, i], axis=1)


        res = np_utils.to_categorical([Service.one_hot_map[name] for name in dataset[:, 4]], 70)
        dataset = np.delete(dataset, 4, axis=1)
        for i in range(70):
            dataset = np.insert(dataset, 4+i, res[:, i], axis=1)


        res = np_utils.to_categorical([flag.one_hot_map[name] for name in dataset[:, 74]], 11)
        dataset = np.delete(dataset, 74, axis=1)
        for i in range(11):
            dataset = np.insert(dataset, 74 + i, res[:, i], axis=1)
        return dataset


    def Number_standardization(dataset):
        #对数缩放
        dataset[:,0] = np.log(dataset[:,0]+1,dtype=float)
        dataset[:,85] = np.log(dataset[:,85]+1,dtype=float)
        dataset[:,86] = np.log(dataset[:,86]+1,dtype=float)

        for i in range(dataset.shape[1]):
            Max = np.max(dataset[:,i])
            Min = np.min(dataset[:,i])
            if Max != Min:
                dataset[:, i] = (dataset[:,i] - Min)/(Max - Min)
        return dataset

    train_x = numerical_coding(train_x)
    train_x = train_x.astype(np.float64)
    train_x = Number_standardization(train_x)

    test_x = numerical_coding(test_x)
    test_x = test_x.astype(np.float64)
    test_x = Number_standardization(test_x)

    print("resampling")
    # 使用 SMOTE 类对数据进行自动重采样
    smote = SVMSMOTE()
    train_x, train_y = smote.fit_resample(train_x, train_y)


    Sampler = RandomOverSampler()
    test_x, test_y = Sampler.fit_resample(test_x, test_y)



    train_y = np_utils.to_categorical(train_y, 5)
    test_y = np_utils.to_categorical(test_y, 5)


    print("data process over,saving to path")

    np.savez(
        os.path.join(path.data_processed_path, 'dataset_train'),
        x=train_x.astype(np.float16), y=train_y,
    )  # 保存训练集
    np.savez(
        os.path.join(path.data_processed_path, 'dataset_test'),
        x=test_x.astype(np.float16), y=test_y,
    )  # 保存测试集

    print("processed data save successfully")
