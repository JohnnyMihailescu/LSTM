import pandas
from time import strftime, gmtime
from pandas import read_csv
from pandas import datetime
from pandas import DataFrame
from pandas import concat
from pandas import Series
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import glob
import math
from matplotlib import pyplot
import numpy as np
from keras.utils import plot_model







def prepare_sequences(x_train, y_train, window_length):
    windows = []
    windows_y = []
    for i, sequence in enumerate(x_train):
        len_seq = len(sequence)
        for window_start in range(0, len_seq - window_length + 1):
            window_end = window_start + window_length
            window = sequence[window_start:window_end]
            windows.append(window)
            windows_y.append(y_train[i])
    return np.array(windows), np.array(windows_y)


N_train = 1000
from numpy.random import choice
one_indexes = choice(a=N_train, size=N_train / 2, replace=False)
X_train[one_indexes, 0] = 1  # very long term memory.