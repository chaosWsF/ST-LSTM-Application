from datetime import datetime
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import ReduceLROnPlateau


output = './figs/'
res_root = './models/'


class lstm:
    def __init__(self, n_time_step, input_dim, n_ahead):
        """
        n_time_step: used in LSTM
        input_dim: #{feature}
        n_ahead: [1, 2, 6, 12, 24, 48] hour(s)
        """
        self.n_time_step = n_time_step
        self.input_dim = input_dim
        self.n_ahead = n_ahead
    
    def setup(self, optimizer='adam'):    # tfa.optimizers.RectifiedAdam
        regressor = Sequential()
        regressor.add(LSTM(units=50, return_sequences=True, input_shape=(self.n_time_step, self.input_dim)))
        regressor.add(Dropout(0))
        regressor.add(Dense(units=1))
        regressor.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

        return regressor
    
    def fit(self, X, Y, lr=1e-7, n_epochs=100, batch_size=32):
        print(datetime.now())

        regressor = self.setup()
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=3, min_lr=lr, cooldown=1)
        # test for a small number of epochs
        history = regressor.fit(X, Y, epochs=n_epochs, batch_size=batch_size, callbacks=[reduce_lr])
        
        print(datetime.now())

        return regressor
    
    def eval(self, X_train, Y_train, X_test, Y_test):
        n_ahead = self.n_ahead
        regressor = self.fit(X_train, Y_train)
        
        Y_pred = regressor.predict(X_test)
        Y_test = Y_test.reshape((Y_test.shape[0], -1))
        Y_pred = Y_pred.reshape((Y_pred.shape[0], -1))

        MAE = np.mean(np.abs(Y_pred - Y_test))
        STD = np.std(Y_pred - Y_test)

        return MAE, STD, Y_pred, Y_test, regressor


def impute_test_data(data, mask):
    """interpolate the missing values with mask value"""
    data = data.replace(mask, float('NaN'))
    data = data.interpolate(method='linear')
    return data


def format_lstm_data(X, Y, n_time_step, stride, n_ahead, freq):
    """
    n_time_step: used in LSTM
    stride: period b/w X_sub
    freq: freq to pick time series
    """
    Xs, Ys = [], []
    period = n_time_step + n_ahead
    i = 0
    while i + period <= len(Y):
        X_sub = X[i:(i+n_time_step):freq, :]
        Y_sub = Y[(i+n_time_step+n_ahead-1), :].reshape((1, -1))
        Xs.append(X_sub)
        Ys.append(Y_sub)
        i += stride
    
    return np.array(Xs), np.array(Ys)


def split_train_test(data, feature_names, target_names, n_ahead, n_time_step, stride, sampling_rate, start, end):
    int_data = impute_test_data(data, 0)    # interpolated data (used for prediction)

    n = len(int_data)
    i_start, i_end = int(n*start), int(n*end)
    train_indices = list(range(i_start)) + list(range(i_end, n))
    train_data = data.iloc[train_indices, :]
    test_data = int_data.iloc[list(range(i_start, i_end)), :]

    X_train = train_data[feature_names].to_numpy()
    Y_train = train_data[target_names].to_numpy()
    X_test = test_data[feature_names].to_numpy()
    Y_test = test_data[target_names].to_numpy()

    X_train, Y_train = format_lstm_data(X_train, Y_train, n_time_step, stride, n_ahead, sampling_rate)
    X_test, Y_test = format_lstm_data(X_test, Y_test, n_time_step, stride, n_ahead, sampling_rate)

    return X_train, Y_train, X_test, Y_test

