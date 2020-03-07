import pickle
import numpy as np


sleep_subsample = pickle.load(open("sleep_subsample.pkl", "rb"))
print(sleep_subsample.shape)


print(list(sleep_subsample.columns))
print(sleep_subsample.head(1))
print(len(sleep_subsample.iloc[0]['eeg']))


'''
cleaned_pandas_dataframe - A pandas dataframe of the format sent to me as sleep_subsample.pkl

VStack EEG, EOG and EMG into a matrix for every unique id

Returns
X_Train - Stacked EEG, EOG and EMG data
Y_Train - Sleep Stage 
'''
def create_stacked_dataset(cleaned_pandas_dataframe):
    X_Train = []
    Y_Train = []
    for i in range(0, cleaned_pandas_dataframe.shape[0]):
        EEG = np.array(cleaned_pandas_dataframe.iloc[i]["eeg"])
        EOG = np.array(cleaned_pandas_dataframe.iloc[i]["eog"])
        EMG = np.array(cleaned_pandas_dataframe.iloc[i]["emg"])

        stacked_data = np.hstack((EEG,EOG,EMG)) #NEED TO GO BACK TO VSTACK IF USING CNN+LSTM HYBRID
        X_Train.append(stacked_data)
        Y_Train.append(cleaned_pandas_dataframe.iloc[i]["stage"])

    return np.array(X_Train), Y_Train


def get_data():
    X_data, Y_data = create_stacked_dataset(sleep_subsample)
    return X_data, Y_data


#X_data, Y_data = create_stacked_dataset(sleep_subsample)


