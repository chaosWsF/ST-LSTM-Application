import sys
import numpy as np
import pandas as pd
from classical_lstm import split_train_test, lstm


modelsPath = "./models"
dataPath = "./data"
stations = ["Old Derry", "MGCC"]

data = pd.read_csv(f"{dataPath}/wide_CRWQ.csv")
feature_names = ['Air Temp_Old Derry', 'Water Temp_Old Derry', 'Chloride Concentration_Old Derry', 'pH_Old Derry', 
                 'Specific Conductivity_Old Derry', 'Turbidity_Old Derry', 
                 'Air Temp_MGCC', 'Water Temp_MGCC', 'Chloride Concentration_MGCC', 'pH_MGCC', 
                 'Specific Conductivity_MGCC', 'Turbidity_MGCC']

n_ahead = int(sys.argv[1])
prefix = sys.argv[2]
assert prefix in ['DO', 'NODO'], "Should be DO or NODO."

## hyperparameters
n_time_step = 5
stride = 7
sampling_rate = 1

p1, p2 = .8, 1



if prefix == 'DO':
    feature_names.append('Dissolved Oxygen_MGCC')

target_names = ['Dissolved Oxygen_Old Derry']

X_train, Y_train, X_test, Y_test = split_train_test(data, feature_names, target_names, 
                                                   n_ahead, n_time_step, stride, sampling_rate, 
                                                   p1, p2)

model = lstm(n_time_step=X_train.shape[1], input_dim=X_train.shape[2], n_ahead=n_ahead)
MAE, STD, Y_pred, Y_test, regressor = model.eval(X_train, Y_train, X_test, Y_test)

regressor.save(f'{modelsPath}/lstm_Old_Derry_{n_ahead}h_{prefix}.h5')
compare_results = np.concatenate((Y_pred, Y_test), axis=1)
compare_results = pd.DataFrame(compare_results)
compare_results.to_csv(f'{modelsPath}/results_lstm_Old_Derry_{n_ahead}h_{prefix}.csv')

df = pd.DataFrame(columns=stations, index=["MAE", "STD"])
df['Old Derry'] = (MAE, STD)



if prefix == 'DO':
    feature_names.append('Dissolved Oxygen_Old Derry')

target_names = ['Dissolved Oxygen_MGCC']

X_train, Y_train, X_test, Y_test = split_train_test(data, feature_names, target_names, 
                                                   n_ahead, n_time_step, stride, sampling_rate, 
                                                   p1, p2)

model = lstm(n_time_step=X_train.shape[1], input_dim=X_train.shape[2], n_ahead=n_ahead)
MAE, STD, Y_pred, Y_test, regressor = model.eval(X_train, Y_train, X_test, Y_test)

regressor.save(f'{modelsPath}/lstm_MGCC_{n_ahead}h_{prefix}.h5')
compare_results = np.concatenate((Y_pred, Y_test), axis=1)
compare_results = pd.DataFrame(compare_results)
compare_results.to_csv(f'{modelsPath}/results_lstm_MGCC_{n_ahead}h_{prefix}.csv')

df['MGCC'] = (MAE, STD)
df.to_csv(f'{modelsPath}/performance_lstm_{n_ahead}h_{prefix}.csv')
