from tensorflow.keras import datasets, layers, models


def create_cnn_model():
    model = models.Sequential()

    model.add(layers.Conv1D(1024, 5, activation='relu', input_shape=(900, 1)))
    model.add(layers.MaxPooling1D(3))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv1D(512, 5, activation='relu'))
    model.add(layers.MaxPooling1D(3))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv1D(256, 5, activation='relu'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Flatten())
    model.add(layers.Dense(2, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    return model


def train_cnn_model(model, X, y, epoch_num):
    epochs = epoch_num

    history = model.fit(
        X,
        y,
        validation_ratio=0.2,
        epochs=epochs)

    return model


def get_cnn_model_predictions(model, X_test):

    return model.predict(X_test)