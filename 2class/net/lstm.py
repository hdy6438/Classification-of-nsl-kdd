from keras.models import Sequential
from keras.layers import Dense, Dropout,LSTM,Conv1D,Flatten,MaxPooling1D
from keras.optimizers import RMSprop
from keras.regularizers import *
from setting import training


def net():
    model = Sequential()  # 创建容器

    #model.add(Conv1D(128,kernel_size=2))

    model.add(LSTM(
        units=128,
        return_sequences=True,
        input_shape=(122, 1),
    ))
    model.add(LSTM(
        units=256,
        return_sequences=True,
        dropout=0.2,
    ))
    model.add(LSTM(
        units=512,
        kernel_regularizer=l2(),
    ))

    # 添加全连接层
    model.add(Dense(512,activation='relu'))
    model.add(Dense(512,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))

    # 添加softmax输出层
    model.add(Dense(2, activation='softmax'))

    # 配置模型
    model.compile(
        loss='categorical_crossentropy',  # 损失函数
        optimizer=RMSprop(learning_rate=training.learning_rate),  # 优化器
        metrics=(['accuracy'])
    )

    model.summary()
    return model
