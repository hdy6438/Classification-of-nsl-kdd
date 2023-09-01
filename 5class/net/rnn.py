from keras.models import Sequential
from keras.layers import Dense, SimpleRNN,Dropout,Input,Conv1D,GlobalMaxPooling1D,Conv2D,MaxPooling2D,GlobalMaxPooling2D
from keras.optimizers import RMSprop
from setting import training


def net():
    model = Sequential()  # 创建容器
    # 添加rnn层

    #model.add(Conv1D(128,kernel_size=2))

    model.add(SimpleRNN(
        units=128,
        return_sequences=True,
        unroll=True,
        input_shape=(122, 1)
    ))
    model.add(SimpleRNN(
        units=256,
        dropout=0.2,
        return_sequences=True,
        unroll=True

    ))

    model.add(SimpleRNN(
        units=512,
        dropout=0.2,
        unroll=True

    ))

    # 添加全连接层
    model.add(Dense(512,activation='relu'))
    model.add(Dense(512,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))

    # 添加softmax输出层
    model.add(Dense(5, activation='softmax'))

    # 配置模型
    model.compile(
        loss='categorical_crossentropy',  # 损失函数
        optimizer=RMSprop(learning_rate=training.learning_rate),  # 优化器
        metrics=(['accuracy'])
    )

    model.summary()
    return model
