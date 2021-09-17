import pandas as pd
import numpy as np


def format_df(df):
    cols = ['time', 'Air Temp', 'Water Temp', 'Chloride Concentration', 'pH', 'Specific Conductivity', 'Turbidity', 'Dissolved Oxygen']
    df = df[cols]
    df = df.fillna(-1)    # replace NA with -1

    var_units = ['(°C)', '(°C)', '(mg/L)', float('NaN'), '(uS/cm)', '(NTU)', '(mg/L)']
    nvar = len(var_units)
    
    df = df.melt(id_vars='time')
    df['variable'] = pd.Categorical(df['variable'], categories=cols, ordered=True)
    df = df.sort_values(by=['time', 'variable'])
    df['unit'] = var_units * (len(df) // nvar)
    df = df.reset_index()
    df = df[['time', 'variable', 'unit', 'value']]

    df['value'] = data_normalize(df['value'])
    
    return df


def format_df_2(df):    # for classical LSTM
    cols = ['time', 'Air Temp', 'Water Temp', 'Chloride Concentration', 'pH', 'Specific Conductivity', 'Turbidity', 'Dissolved Oxygen']
    df = df[cols]
    df = df.fillna(-1)    # replace NA with -1
    df[cols[1:]] = df[cols[1:]].apply(data_normalize, axis=0)
    return df


def data_normalize(d):
    """
    Data normalization:
        using log function and skip the masked data with -1
        0 values are replaced by 1e-5 in order to avoid nan value in log
    """
    new_d = []
    for val in d:
        if val == -1:
            new_d.append(val)
        else:
            if val > 0:
                norm = np.log(val)
            elif val < 0:
                norm = -np.log(-val)
            else:
                norm = 1e-5
            
            new_d.append(norm)
    
    return new_d


dataRoot = "./data"
df = pd.read_excel(f"{dataRoot}/Credit River Water Quality Data.xlsx", sheet_name="Hourly Water Quality")
df1, df2 = df.iloc[2:,:8], df.iloc[2:,8:]    # Old Derry, MGCC
df1 = df1.rename(columns={'Unnamed: 0': 'time'})
df2 = df2.rename(columns=lambda s: s.replace('.1', ''))
df2['time'] = df1['time']
spData = {'Old Derry': format_df(df1), 'MGCC': format_df(df2)}

for sensor in spData:
    spData[sensor].to_csv(f"{dataRoot}/{sensor}.csv", index=False)

df1 = format_df_2(df1)
df2 = format_df_2(df2)
data = df1.merge(df2, on='time', suffixes=['_Old Derry', '_MGCC'])
data.to_csv(f"{dataRoot}/wide_CRWQ.csv", index=False)
