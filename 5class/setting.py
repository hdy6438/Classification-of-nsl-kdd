from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard


class path:
    data_path = "data"
    data_processed_path = "data_process"
    rnn_model_path = 'model/rnn/model.h5'
    rnn_log_path = 'model/rnn/logs'
    dnn_model_path = 'model/dnn/model.h5'
    dnn_log_path = 'model/dnn/logs'
    lstm_model_path = 'model/lstm/model.h5'
    lstm_log_path = 'model/lstm/logs'

class training:
    batch_size = 2048
    epochs = 300
    learning_rate = 1e-4


def get_callbacks(net_name):
    if net_name == "rnn":
        model_path = path.rnn_model_path
        log_dir = path.rnn_log_path
    elif net_name == "dnn":
        model_path = path.dnn_model_path
        log_dir = path.dnn_log_path
    elif net_name == "lstm":
        model_path = path.lstm_model_path
        log_dir = path.lstm_log_path
    else:
        raise ValueError("net name error")
    return [
        # 模型检查,只保留最好的
        ModelCheckpoint(
            filepath=model_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
            period=1,

        ),
        # 当指标不再改善时减低学习率
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=15,
            verbose=1,
            mode='auto',
            min_delta=0,
            cooldown=15,
            min_lr=0,
        ),
        # 模型可视化
        TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_images=True,
            write_graph=True,
            write_grads=True,
        )

    ]




