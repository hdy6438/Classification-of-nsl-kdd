import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import make_interp_spline

if __name__ == "__main__":

    dnn_5 =  np.load("../5class/model/dnn/info.npy", allow_pickle=True).item()
    dnn_5_loss = dnn_5['loss']
    dnn_5_val_loss = dnn_5['val_loss']
    dnn_5_accuracy = dnn_5['accuracy']
    dnn_5_val_accuracy = dnn_5['val_accuracy']

    rnn_5 =  np.load("../5class/model/rnn/info.npy", allow_pickle=True).item()
    rnn_5_loss = rnn_5['loss']
    rnn_5_val_loss = rnn_5['val_loss']
    rnn_5_accuracy = rnn_5['accuracy']
    rnn_5_val_accuracy = rnn_5['val_accuracy']




    fig = plt.figure()  # 新建画布
    plt.title("Comparison of Train Accuracy")  # 设置标题
    plt.plot(np.linspace(1, 300, 100), make_interp_spline(list(range(1,301)), dnn_5_accuracy)(np.linspace(1, 300, 100)), label="dnn")
    plt.plot(np.linspace(1, 300, 100), make_interp_spline(list(range(1,301)), rnn_5_accuracy)(np.linspace(1, 300, 100)), label="rnn")
    plt.xlabel("Epoch")  # 设置x坐标标签
    plt.ylabel("Accuracy")  # 设置y坐标标签
    plt.legend(loc="lower left")  # 显示图例
    plt.show()  # 显示图像
    fig.savefig("5_accuracy_train.png")  # 保存图像
    plt.close(fig)  # 清理内存

    fig = plt.figure()  # 新建画布
    plt.title("Comparison of Test Accuracy")  # 设置标题
    plt.plot(np.linspace(1, 300, 100), make_interp_spline(list(range(1, 301)), dnn_5_val_accuracy)(np.linspace(1, 300, 100)), label="dnn")
    plt.plot(np.linspace(1, 300, 100), make_interp_spline(list(range(1, 301)), rnn_5_val_accuracy)(np.linspace(1, 300, 100)), label="rnn")
    plt.xlabel("Epoch")  # 设置x坐标标签
    plt.ylabel("Accuracy")  # 设置y坐标标签
    plt.legend(loc="lower left")  # 显示图例
    plt.show()  # 显示图像
    fig.savefig("5_accuracy_test.png")  # 保存图像
    plt.close(fig)  # 清理内存




    fig = plt.figure()  # 新建画布
    plt.title("Comparison of Train loss")  # 设置标题
    plt.plot(np.linspace(1, 300, 100), make_interp_spline(list(range(1, 301)), dnn_5_loss)(np.linspace(1, 300, 100)), label="dnn")
    plt.plot(np.linspace(1, 300, 100), make_interp_spline(list(range(1, 301)), rnn_5_loss)(np.linspace(1, 300, 100)), label="rnn")
    plt.xlabel("Epoch")  # 设置x坐标标签
    plt.ylabel("Loss")  # 设置y坐标标签
    plt.legend(loc="lower left")  # 显示图例
    plt.show()  # 显示图像
    fig.savefig("5_loss_train.png")  # 保存图像
    plt.close(fig)  # 清理内存

    fig = plt.figure()  # 新建画布
    plt.title("Comparison of Test loss")  # 设置标题
    plt.plot(np.linspace(1, 300, 100), make_interp_spline(list(range(1, 301)), dnn_5_val_loss)(np.linspace(1, 300, 100)), label="dnn")
    plt.plot(np.linspace(1, 300, 100), make_interp_spline(list(range(1, 301)), rnn_5_val_loss)(np.linspace(1, 300, 100)), label="rnn")
    plt.xlabel("Epoch")  # 设置x坐标标签
    plt.ylabel("Loss")  # 设置y坐标标签
    plt.legend(loc="lower left")  # 显示图例
    plt.show()  # 显示图像
    fig.savefig("5_loss_test.png")  # 保存图像
    plt.close(fig)  # 清理内存




