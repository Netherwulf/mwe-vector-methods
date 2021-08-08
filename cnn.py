import glob

from tensorflow.keras import datasets, layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


def create_cnn_model():
    model = models.Sequential()

    model.add(layers.Conv1D(1024, 5, activation='relu', input_shape=(900, 1)))
    model.add(layers.MaxPooling1D(3))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv1D(512, 5, activation='relu'))
    model.add(layers.MaxPooling1D(3))
    model.add(layers.Dropout(0.2))
    #model.add(layers.Conv1D(256, 5, activation='relu'))
    #model.add(layers.MaxPooling1D(3))
    #model.add(layers.Dropout(0.2))
    model.add(layers.Conv1D(256, 5, activation='relu')) # 256
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Flatten())
    model.add(layers.Dense(2, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    return model

def find_best_checkpoint():
    curr_loss = 0

    for file in glob.glob("models/*.hdf5"):

        if int(file.split('.')[-2]) > curr_loss:
            best_checkpoint = file
            curr_loss = int(file.split('.')[-2])

    return best_checkpoint

def train_cnn_model(model, X, y, epoch_num):
    epochs = epoch_num

    # callback = EarlyStopping(monitor='loss', patience=10)

    checkpoint_filepath = 'models/checkpoint_epoch_{epoch:04d}_val_{val_accuracy:.4f}.hdf5'

    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    # Model weights are saved at the end of every epoch, if it's the best seen
    # so far.
    history = model.fit(
        X,
        y,
        validation_split=0.2,
        epochs=epochs,
        callbacks=[model_checkpoint_callback])

    best_checkpoint_filepath = find_best_checkpoint()

    # The model weights (that are considered the best) are loaded into the model.
    model.load_weights(best_checkpoint_filepath)

    return model


def get_cnn_model_predictions(model, X_test):

    return model.predict(X_test)


def get_cnn_model_pred(X_train, y_train, X_test, eval_only=False, model_path=None):
    cnn_model = create_cnn_model()

    if eval_only:
        cnn_model.load_weights(model_path)

    else:
        cnn_model = train_cnn_model(cnn_model, X_train, y_train, 1000)

    y_pred = get_cnn_model_predictions(cnn_model, X_test)

    return y_pred
