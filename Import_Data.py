import pickle
import pandas
import numpy as np


sleep_subsample = pickle.load(open("sleep_subsample.pkl", "rb"))
print(sleep_subsample.shape)


print(list(sleep_subsample.columns))
print(sleep_subsample.head(1))
print(len(sleep_subsample.iloc[0]['eeg']))


'''
cleaned_pandas_dataframe - A pandas dataframe of the format sent to me as sleep_subsample.pkl

VStack EEG, EOG and EMG into a matrix for every unique id
'''
def create_stacked_dataset(cleaned_pandas_dataframe):
    X_Train = []
    Y_Train = []
    for i in range(0, cleaned_pandas_dataframe.shape[0]):
        EEG = np.array(cleaned_pandas_dataframe.iloc[i]["eeg"])
        EOG = np.array(cleaned_pandas_dataframe.iloc[i]["eog"])
        EMG = np.array(cleaned_pandas_dataframe.iloc[i]["emg"])

        stacked_data = np.vstack((EEG,EOG,EMG))
        X_Train.append(stacked_data)
        Y_Train.append(cleaned_pandas_dataframe.iloc[i]["stage"])

    return X_Train, Y_Train


