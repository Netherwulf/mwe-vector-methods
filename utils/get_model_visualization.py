import sys

from cnn import create_cnn_model

from tensorflow.keras import models
from tensorflow.keras.utils import plot_model


def load_model(filepath):
    return models.load_model(filepath)


def visualize_model(filepath=None):
    if filepath is None:
        model = create_cnn_model(input_shape=(900, 1))

    else:
        model = load_model(filepath)

    plot_model(model, to_file='model_visualization.png', show_shapes=False,
        show_layer_names=False, rankdir='TB', expand_nested=False, dpi=300)

def main(args):
    if len(args) == 0:
        visualize_model(None)

    else:
        for filepath in args:
            visualize_model(filepath)


if __name__ == '__main__':
    main(sys.argv[1:])
