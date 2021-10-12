import glob
import os

import numpy as np

from tensorflow import keras
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


def create_cnn_model(input_shape=(900, 1)):
    model = models.Sequential()

    model.add(layers.Conv1D(1024, 5, activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling1D(3))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv1D(512, 5, activation='relu'))
    model.add(layers.MaxPooling1D(3))
    model.add(layers.Dropout(0.2))
    # model.add(layers.Conv1D(256, 5, activation='relu'))
    # model.add(layers.MaxPooling1D(3))
    # model.add(layers.Dropout(0.2))
    model.add(layers.Conv1D(256, 5, activation='relu'))  # 256
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Flatten())
    model.add(layers.Dense(2, activation='softmax'))

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                  loss='binary_crossentropy',
                  metrics=['AUC', 'accuracy', keras.metrics.Precision(), keras.metrics.Recall()])
    model.summary()

    return model


def find_best_checkpoint(dir_name):
    curr_loss = 0

    for filepath in glob.glob(f"{dir_name}/*.hdf5"):

        if int(filepath.split('.')[-2]) > curr_loss:
            best_checkpoint = filepath
            curr_loss = int(filepath.split('.')[-2])

    return best_checkpoint


def get_class_weights(train_labels):
    unique, counts = np.unique(train_labels, return_counts=True)

    class_counts = dict(zip(unique, counts))

    class_weights = dict()

    for class_label in class_counts.keys():
        class_weights[class_label] = train_labels.shape[0] / (len(class_counts.keys()) * class_counts[class_label])

    return class_weights


def get_dir_num():
    last_dir_num = 0

    if len(glob.glob('models_*')):
        return last_dir_num

    else:
        for filepath in glob.glob('models_*'):
            if int(filepath.split('_')[-1]) > last_dir_num:
                last_dir_num = int(filepath.split('_')[-1])

        return last_dir_num + 1


def train_cnn_model(model, X, y, epoch_num):
    epochs = epoch_num

    # callback = EarlyStopping(monitor='loss', patience=10)
    dir_name = f'models_{get_dir_num()}'

    os.mkdir(dir_name)

    checkpoint_filepath = dir_name + '/checkpoint_epoch_{epoch:04d}_val_{val_auc:.4f}.hdf5'

    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_auc',
        mode='max',
        save_best_only=True)

    # Model weights are saved at the end of every epoch, if it's the best seen
    # so far.
    history = model.fit(
        X,
        y,
        batch_size=64,
        validation_split=0.2,
        epochs=epochs,
        class_weight = get_class_weights(y),
        callbacks=[model_checkpoint_callback])

    best_checkpoint_filepath = find_best_checkpoint(dir_name)

    print(f'Current directory index: {dir_name}')

    # The model weights (that are considered the best) are loaded into the model.
    model.load_weights(best_checkpoint_filepath)

    return model


def get_cnn_model_predictions(model, X_test):
    return model.predict(X_test)


def get_cnn_model_pred(X_train, y_train, X_test, eval_only=False, model_path=None, input_shape=(900, 1)):
    cnn_model = create_cnn_model(input_shape=input_shape)

    if eval_only:
        cnn_model.load_weights(model_path)

    else:
        cnn_model = train_cnn_model(cnn_model, X_train, y_train, 15)  # 1000 epochs

    y_pred = get_cnn_model_predictions(cnn_model, X_test)

    return y_pred
