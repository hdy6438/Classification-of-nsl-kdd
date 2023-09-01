import json
import os
import shutil

import numpy as np

from setting import path, training,get_callbacks
from tool.data_loader import DataGenerator
from tool.draw import draw

if __name__ == "__main__":

    net_name = "dnn"

    if net_name == "rnn":
        from net.rnn import net
    elif net_name == "dnn":
        from net.dnn import net
    elif net_name == "lstm":
        from net.lstm import net
    else:
        raise ValueError("net name error")

    model = net() #实例化神经网络

    dataset_train = DataGenerator(
        os.path.join(path.data_processed_path, "dataset_train.npz"),
        batch_size=training.batch_size,
        net_name = net_name,
    )  # 加载训练数据

    dataset_test = DataGenerator(
        os.path.join(path.data_processed_path, "dataset_test.npz"),
        batch_size=training.batch_size,
        net_name = net_name,
    )  # 加载测试数据

    if os.path.exists(os.path.join("model",net_name)):
        shutil.rmtree(os.path.join("model",net_name))  # 清空model文件夹

    history = model.fit(dataset_train, validation_data=dataset_test, batch_size=training.batch_size,
                        epochs=training.epochs, callbacks=get_callbacks(net_name))  # 开始训练

    # 画图
    loss = [dict(data=history.history['loss'], label="loss"), dict(data=history.history['val_loss'], label="val_loss")]
    acc = [dict(data=history.history['accuracy'], label="accuracy"),
           dict(data=history.history['val_accuracy'], label="val_accuracy")]
    lr = [dict(data=history.history['lr'], label="lr")]

    np.save(os.path.join("model",net_name,"info"),history.history)

    draw(list(range(training.epochs)), "loss and val_loss", loss, "loss",net_name)
    draw(list(range(training.epochs)), "acc and val_accuracy", acc, "acc",net_name)
    draw(list(range(training.epochs)), "lr", lr, "lr",net_name)
