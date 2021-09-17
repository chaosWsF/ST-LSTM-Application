import os
import sys
import numpy as np
import pandas as pd
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"    # if using CPU (faster than GPU)
from model import sp_lstm


modelsPath = "./models"
dataPath = "./data"
stations = ["Old Derry", "MGCC"]

crwq = []
for station in stations:
    df = pd.read_csv(f"{dataPath}/{station}.csv")
    crwq.append(df['value'].to_list())


n_ahead = int(sys.argv[1])

## hyperparameters
train_hour = 7

p1, p2 = .8, 1
n_features = 6
predict_hour = 1
stride = n_features + predict_hour
predict_position = n_ahead * stride + (n_features - train_hour)

crwq_error, crwq_sd = sp_lstm(crwq, n_ahead, train_hour, predict_hour, predict_position, stride, p1, p2)
df = pd.DataFrame(columns=stations, index=["MAE", "STD"])
df.loc["MAE", :] = crwq_error
df.loc["STD", :] = crwq_sd
df.to_csv(f'{modelsPath}/performance_sp_lstm_{n_ahead}h.csv')
