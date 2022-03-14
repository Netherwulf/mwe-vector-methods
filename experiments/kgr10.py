import argparse
import csv
import datetime
import os
import pickle as pkl
import re
import statistics
import sys

import numpy as np
import pandas as pd

from models.cnn import get_cnn_model_pred
from models.logistic_regression import get_lr_model_pred
from models.random_forest import get_rf_model_pred

from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from scipy import stats as s
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow import one_hot, config


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--undersampled',
                        help='use undersampled version of the dataset',
                        action='store_true')

    args = parser.parse_args()

    return args
