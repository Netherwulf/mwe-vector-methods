from sklearn.linear_model import LogisticRegression


def create_lr_model():

    return LogisticRegression(random_state=0, max_iter=2000, verbose=1)


def train_lr_model(model, X, y):
    model.fit(X, y)

    return model


def get_lr_model_predictions(model, X_test):

    return model.predict(X_test)
