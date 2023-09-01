import os
from keras.models import load_model
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from setting import path

if __name__ == "__main__":
    model = load_model(path.rnn_model_path)  # 加载模型

    dataset_test = np.load(os.path.join(path.data_processed_path, "dataset_test.npz"))  # 加载测试数据
    x = dataset_test['x']  # 提取数据
    y = dataset_test['y']  # 提取标签

    x = x.reshape((x.shape[0],-1,1)) #dnn时需要注释掉

    predict = model.predict(x)  # 将数据集输入模型
    y_test_pred = np.argmax(predict, axis=1)
    y_test_true = np.argmax(y, axis=1)

    labels = ["dos", "r2l","probe","normal","u2r"]  # 设置混淆矩阵标签

    cm = confusion_matrix(y_true=y_test_true, y_pred=y_test_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)  # 画混淆矩阵
    disp.plot(cmap=plt.cm.Blues)

    plt.show()  # 显示混淆矩阵
