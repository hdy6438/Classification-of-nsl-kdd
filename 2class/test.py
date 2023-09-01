import json
import os
from keras.models import load_model
from tool.data_loader import DataGenerator
from setting import path, training

if __name__ == "__main__":
    net_name = "rnn"

    if net_name == "rnn":
        model = load_model(path.rnn_model_path)  # 加载模型

    elif net_name == "dnn":
        model = load_model(path.dnn_model_path)  # 加载模型

    else:
        raise ValueError("net name error")

    dataset_train = DataGenerator(
        os.path.join(path.data_processed_path, "dataset_train.npz"),
        batch_size=training.batch_size,
        net_name=net_name
    )  # 加载训练集

    dataset_test = DataGenerator(
        os.path.join(path.data_processed_path, "dataset_test.npz"),
        batch_size=training.batch_size,
        net_name=net_name

    )  # 加载测试集

    train_test = model.evaluate(dataset_train)  # 基于train_dataset评估模型
    val_test = model.evaluate(dataset_test)  # 基于test_dataset评估模型

    # 保存结果
    data = {
        "train": dict(loss=train_test[0], acc=train_test[1]),
        "val": dict(loss=val_test[0], acc=val_test[1])
    }

    with open(os.path.join("model",net_name,"test.json"), mode="w") as f:
        json.dump(data, f)
        f.close()
