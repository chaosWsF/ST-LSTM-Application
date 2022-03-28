import joblib
import numpy as np
from pathlib import Path
from models import *

cache_dir = Path('cache')
x_train = np.load(cache_dir / Path('X_train.npy'))
y_train = np.load(cache_dir / Path('y_train.npy'))
x_test = np.load(cache_dir / Path('X_test.npy'))
y_test = np.load(cache_dir / Path('y_test.npy'))
scaler = joblib.load(cache_dir / Path('scaler.gz'))

## TODO: with upstream or only downstream
working_dir = Path('results_with_up')
# working_dir = Path('results_only_down')

working_dir.mkdir(exist_ok=True)

EPCOHS = 1
# EPOCHS = 30    # TODO
BATCH_SIZE = 32
LR = 1E-3

print('=====START=====')
run_lstm(LR, BATCH_SIZE, EPCOHS, scaler, x_train, y_train, x_test, y_test, working_dir)
print('=====LSTM done=====')
run_cnnlstm(LR, BATCH_SIZE, EPCOHS, scaler, x_train, y_train, x_test, y_test, working_dir)
print('=====CNN LSTM done=====')
run_convlstm(LR, BATCH_SIZE, EPCOHS, scaler, x_train, y_train, x_test, y_test, working_dir)
print('=====ConvLSTM done=====')
run_seq2seq(LR, BATCH_SIZE, EPCOHS, scaler, x_train, y_train, x_test, y_test, working_dir)
print('=====Seq2Seq done=====')
run_wavenet(LR, BATCH_SIZE, EPCOHS, scaler, x_train, y_train, x_test, y_test, working_dir)
print('=====Wavenet done=====')
