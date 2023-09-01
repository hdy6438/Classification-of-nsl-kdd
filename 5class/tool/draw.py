import os.path
import matplotlib.pyplot as plt


def draw(x, title, data, ylabel,net_name):
    fig = plt.figure()  # 新建画布
    plt.title(title)  # 设置标题
    for y in data:  # 绘制曲线
        plt.plot(x, y['data'], label=y['label'])
    plt.xlabel("epochs")  # 设置x坐标标签
    plt.ylabel(ylabel)  # 设置y坐标标签
    plt.legend(loc="lower left")  # 显示图例
    plt.show()  # 显示图像
    fig.savefig(os.path.join("model",net_name, title))  # 保存图像
    plt.close(fig)  # 清理内存

