import fasttext


def load_fasttext(model_path):
    model = fasttext.load_model(model_path)

    return model
