import pickle

from sklearn.linear_model import LogisticRegression


def create_lr_model():

    return LogisticRegression(random_state=0, max_iter=2000, verbose=1)


def train_lr_model(model, X, y):
    model.fit(X, y)

    return model


def save_lr_model(model, filename):
    pickle.dump(model, open(filename, 'wb'))


def load_lr_model(filename):
    loaded_model = pickle.load(open(filename, 'rb'))

    return loaded_model


def get_lr_model_predictions(model, X_test):

    return model.predict(X_test)


def get_lr_model_pred(X_train, y_train, X_test):
    lr_model = create_lr_model()

    lr_model = train_lr_model(lr_model, X_train, y_train)

    save_lr_model(lr_model, 'lr_model.pkl')

    y_pred = get_lr_model_predictions(lr_model, X_test)

    return y_pred
