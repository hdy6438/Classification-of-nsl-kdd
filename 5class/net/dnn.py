from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.optimizers import RMSprop
from setting import training


def net():
    model = Sequential()  # 创建容器
    # 添加dnn层
    model.add(Dense(32, input_shape=(122,)))
    model.add(Dense(32))
    model.add(Dropout(0.5))

    model.add(Dense(64))
    model.add(Dense(64))
    model.add(Dropout(0.5))

    model.add(Dense(5, activation='softmax'))

    # 配置模型
    model.compile(
        loss='categorical_crossentropy',  # 损失函数
        optimizer=RMSprop(learning_rate=training.learning_rate),  # 优化器
        metrics=(['accuracy'])
    )

    model.summary()
    return model
