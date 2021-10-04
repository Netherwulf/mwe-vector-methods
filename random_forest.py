import pickle

from sklearn.ensemble import RandomForestClassifier


def create_rf_model():

    return RandomForestClassifier(random_state=0, max_depth=13,
                                  verbose=1, class_weight='balanced')


def train_rf_model(model, X, y):
    model.fit(X, y)

    return model


def save_rf_model(model, filename):
    pickle.dump(model, open(filename, 'wb'))


def load_rf_model(filename):
    loaded_model = pickle.load(open(filename, 'rb'))

    return loaded_model


def get_rf_model_predictions(model, X_test):

    return model.predict(X_test)


def get_rf_model_pred(X_train, y_train, X_test):
    rf_model = create_rf_model()

    rf_model = train_rf_model(rf_model, X_train, y_train)

    save_rf_model(rf_model, 'rf_model.pkl')

    y_pred = get_rf_model_predictions(rf_model, X_test)

    return y_pred
