import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import make_interp_spline

if __name__ == "__main__":

    dnn_2 =  np.load("../2class/model/dnn/info.npy", allow_pickle=True).item()
    dnn_2_loss = dnn_2['loss']
    dnn_2_val_loss = dnn_2['val_loss']
    dnn_2_accuracy = dnn_2['accuracy']
    dnn_2_val_accuracy = dnn_2['val_accuracy']

    rnn_2 =  np.load("../2class/model/rnn/info.npy", allow_pickle=True).item()
    rnn_2_loss = rnn_2['loss']
    rnn_2_val_loss = rnn_2['val_loss']
    rnn_2_accuracy = rnn_2['accuracy']
    rnn_2_val_accuracy = rnn_2['val_accuracy']




    fig = plt.figure()  # 新建画布
    plt.title("Comparison of Train Accuracy")  # 设置标题
    plt.plot(np.linspace(1, 300, 100), make_interp_spline(list(range(1,301)), dnn_2_accuracy)(np.linspace(1, 300, 100)), label="dnn")
    plt.plot(np.linspace(1, 300, 100), make_interp_spline(list(range(1,301)), rnn_2_accuracy)(np.linspace(1, 300, 100)), label="rnn")
    plt.xlabel("Epoch")  # 设置x坐标标签
    plt.ylabel("Accuracy")  # 设置y坐标标签
    plt.legend(loc="lower left")  # 显示图例
    plt.show()  # 显示图像
    fig.savefig("2_accuracy_train.png")  # 保存图像
    plt.close(fig)  # 清理内存

    fig = plt.figure()  # 新建画布
    plt.title("Comparison of Test Accuracy")  # 设置标题
    plt.plot(np.linspace(1, 300, 100), make_interp_spline(list(range(1, 301)), dnn_2_val_accuracy)(np.linspace(1, 300, 100)), label="dnn")
    plt.plot(np.linspace(1, 300, 100), make_interp_spline(list(range(1, 301)), rnn_2_val_accuracy)(np.linspace(1, 300, 100)), label="rnn")
    plt.xlabel("Epoch")  # 设置x坐标标签
    plt.ylabel("Accuracy")  # 设置y坐标标签
    plt.legend(loc="lower left")  # 显示图例
    plt.show()  # 显示图像
    fig.savefig("2_accuracy_test.png")  # 保存图像
    plt.close(fig)  # 清理内存




    fig = plt.figure()  # 新建画布
    plt.title("Comparison of Train loss")  # 设置标题
    plt.plot(np.linspace(1, 300, 100), make_interp_spline(list(range(1, 301)), dnn_2_loss)(np.linspace(1, 300, 100)), label="dnn")
    plt.plot(np.linspace(1, 300, 100), make_interp_spline(list(range(1, 301)), rnn_2_loss)(np.linspace(1, 300, 100)), label="rnn")
    plt.xlabel("Epoch")  # 设置x坐标标签
    plt.ylabel("Loss")  # 设置y坐标标签
    plt.legend(loc="lower left")  # 显示图例
    plt.show()  # 显示图像
    fig.savefig("2_loss_train.png")  # 保存图像
    plt.close(fig)  # 清理内存

    fig = plt.figure()  # 新建画布
    plt.title("Comparison of Test loss")  # 设置标题
    plt.plot(np.linspace(1, 300, 100), make_interp_spline(list(range(1, 301)), dnn_2_val_loss)(np.linspace(1, 300, 100)), label="dnn")
    plt.plot(np.linspace(1, 300, 100), make_interp_spline(list(range(1, 301)), rnn_2_val_loss)(np.linspace(1, 300, 100)), label="rnn")
    plt.xlabel("Epoch")  # 设置x坐标标签
    plt.ylabel("Loss")  # 设置y坐标标签
    plt.legend(loc="lower left")  # 显示图例
    plt.show()  # 显示图像
    fig.savefig("2_loss_test.png")  # 保存图像
    plt.close(fig)  # 清理内存




