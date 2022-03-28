import pandas as pd


def read_file(path):
    '''
    Returns the dataframe which is read from the excel file present in the path specified. 
    
    Parameters:
        path (str) : The path of the file
    
    Returns:
        df (float) : The dataframe which is created after reading the file.
    '''
    df= pd.read_csv(path)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    return df


## Specifying the source path
filename_upstream = r'oldderry.csv'
filename_downstream = r'MGCC.csv'

## Reading the file
df_up = read_file(filename_upstream)
df_down = read_file(filename_downstream)

## Dropping the columns that are not needed such as Discharge, since all the values are NaN
df_down.drop(columns=['Discharge'], inplace=True)

## Splitting the Time Hour column into year,month,date, hour columns
year_values = []
month_values = []
date_values = []
time_values = []

for i in df_up['Time Hour']:
    year, month, date_time = i.split('/')
    date, time = date_time.split(' ')
    time = time[:-3]
    year_values.append(year)
    month_values.append(month)
    date_values.append(date)
    time_values.append(time)

## adding the new column and dropping the old Time Hour column in df dataFrame
df_up['Year'] = year_values
df_up['Month'] = month_values
df_up['Date'] = date_values
df_up['Time'] = time_values

## dropping "Time Hour" column
df_up.drop(columns=['Time Hour'], inplace=True)

## adding the new column and dropping the old Time Hour column in df dataFrame
df_down['Year'] = year_values
df_down['Month'] = month_values
df_down['Date'] = date_values
df_down['Time'] = time_values

## dropping "Time Hour" column
df_down.drop(columns=['Time Hour'], inplace=True)

df_up.rename(columns={"Air Temp": "Air_Temp_Up", "Water Temp": "Water_Temp_Up", "Chloride Concentration": "Chloride_Up", "pH": "pH_Up", "Specific Conductivity": "Cond_Up", "Turbidity": "Turbidity_Up", "Dissolved Oxygen": "DO_Up",}, inplace=True)
df_down.rename(columns={"Air Temp": "Air_Temp_Dn", "Water Temp": "Water_Temp_Dn", "Chloride Concentration": "Chloride_Dn", "pH": "pH_Dn", "Specific Conductivity": "Cond_Dn", "Turbidity": "Turbidity_Dn", "Dissolved Oxygen": "DO_Dn",}, inplace=True)

## save cleaned datasets
df_up.to_csv(r'upstream.csv', index=False)
df_down.to_csv(r'downstream.csv', index=False)
