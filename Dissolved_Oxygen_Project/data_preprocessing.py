import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path


def create_dataset(X, y, time_steps, ts_range):
    '''
    Returns the prepared data based on the lag and look ahead
    
    Parameters:
        X          (float): The independent variables of the data
        y          (float): The dependent variables of the data
        time_steps (int)  : The lag that is being used to lookback
        ts_range   (int)  : The lookahead for the data
    
    Returns:
        Xs (float) : The numpy array of the input variable
        ys (float) : The numpy array of the output variable 
    '''
    Xs, ys = [], []
    for i in range(len(X) - time_steps - ts_range):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.values[(i + time_steps):(i + time_steps + ts_range),0])
    
    return np.array(Xs), np.array(ys)


def splitter(df,output,lag,duration,ts):
    '''
    Returns the training and testing data
    
    Parameters:
        df (float): The whole dataframe containing the independent and dependent variables
        output(str): The output variable 
        lag (int): The lag that needs to be applied for the data
        duration (int): The duration that is being considered as output
        ts (float): The percentage of training data
    
    Returns:
        x_train (float): The training data of independent variable 
        x_test (float): The testing data of independent variable
        y_train (float): The training data of the depenedent variable 
        y_test (float): The testing data of the dependent variable 
    '''
    assert (0. <= ts <= 1.)
    train_size = int(len(df) * ts)
    test_size = len(df) - train_size
    train, test = df.iloc[0:train_size], df.iloc[train_size:]
    scaler,scaler_single = MinMaxScaler(feature_range=(0, 1)), MinMaxScaler(feature_range=(0, 1))

    scaler.fit(train)
    scaler_single.fit(train[[output]])

    train_scaled = pd.DataFrame(scaler.transform(train), columns=[df.columns])
    test_scaled = pd.DataFrame(scaler.transform(test), columns=[df.columns])

    df_train = train_scaled.copy(deep=True)
    df_test = test_scaled.copy(deep=True)

    x_train,y_train = create_dataset(df_train, df_train[[output]], lag, duration)
    x_test, y_test = create_dataset(df_test, df_test[[output]], lag, duration)

    return x_train,x_test,y_train,y_test,scaler_single


df_up = pd.read_csv(r'upstream.csv')
df_down = pd.read_csv(r'downstream.csv')

## combine up- and down-stream data with lag = 0 hour
df_total = pd.merge(df_up, df_down)

## change this feature combination list to do different combination of feature testing
feat_cols = [
    'Water_Temp_Up', 'Cond_Up', 'DO_Up',    # TODO: Comment this line to use downstream only
    'Water_Temp_Dn', 'Cond_Dn'
]
target_col = 'DO_Dn'

## Change the lag, duration variable values according to experiment value taken
## EPOCHS = 5 (taken for demonstration of code working, later we can change it 200 or higher epochs)
lag = 6
duration = 6
EPOCHS = 30
BATCH_SIZE = 32
LR = 1E-3
train_size = 0.8

## Creating the training and testing data
x_train,x_test,y_train,y_test,scaler = splitter(df_total[feat_cols + [target_col]], target_col, lag, duration, train_size)

## save preprocessed data
cache_dir = Path('cache')
cache_dir.mkdir(exist_ok=True)
np.save(cache_dir / Path('X_train.npy'), x_train)
np.save(cache_dir / Path('y_train.npy'), y_train)
np.save(cache_dir / Path('X_test.npy'), x_test)
np.save(cache_dir / Path('y_test.npy'), y_test)
