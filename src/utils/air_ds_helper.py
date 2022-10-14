
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# Imports
import math
import numpy as np
import pandas as pd
import pickle
import gzip
import os
import copy
def load_air_ds(path, 
                window= 168, 
                mode='train',
                col_index= [0, 1, 2], 
                stride= None, 
                label_type="last",
                start= None,
                end= None,
                resample= None,
                fillna= True, 
                verbose=False,
                *args,
                **kwargs):
    """
    Creates a dataset object from the given time series data.

    Parameters
    ----------
    path : str or Path-like
        Path to data.
    sensor : str columns name. If None all of columns selected
    resample : str or pandas offset object
        frequency to use to resample data.
        Must be one of pandas date offset strings or corresponding objects.
    fillna : bool, default True
        If True, fill NaNs with 0 after resampling.
    col_index : list of str, optional
        List of columns to use. Default is `None`, which means all columns.
    window : str or pandas offset object, default "168"
        Time window to use to create timeseries from the dataset.
        Must be one of pandas date offset strings or corresponding objects.
    stride : str or pandas offset object, default None
        Offset between two consecutive timeseries.
        Must be one of pandas date offset strings or corresponding objects.
    label_type : str, default "last"
        Must be one of `["last", "all"]`.
        If `last`, the time series is labelled looking only at the last timestep.
        If `all`, all timepoints in the timeseries are labelled.
    """
    # read data
    df = pd.read_csv(path, usecols=col_index, header=0,
                            names=col_index,
                            encoding="utf-8")
    df = df.set_index(pd.to_datetime(df[col_index[0]], origin='unix', unit="ns")).drop(columns=[col_index[0]])
    
    # resample dataset
    if resample is not None:
        df = df.resample(resample).mean()
        if fillna:
            df.fillna(0, inplace=True)
    
    if start is None:
        start = df.index[0]
    if end is None:
        end = df.index[-1]
    
    # make X, y dataset
    if label_type not in ["last", "all"]:
        print("label must be one of ['last', 'all']. Setting to 'last'.")
        label_type = "last"
    
    stride = window if stride is None else stride
    
    if verbose:
        print(f"Window: {window}, Stride: {stride}")
    X, y = make_sliding_windows(
        df, window, stride, label_type)
    return X, y

def make_sliding_windows(df:pd.DataFrame, 
                        window:int, 
                        stride:int= None,
                        label_type= "last"):
    """
    Create sliding windows from a DataFrame and optionally assigns a label based on `anomaly`.
    The label is given looking at the last timestamp of each time series.

    Parameters
    ----------
    df : pandas DataFrame
        Dataset with time index.
    window : int, Size of the temporal window.
    stride : int, Stride between consecutive temporal windows.
    label_type : str, default "last"
        Must be one of `["last", "all"]`.
        If `last`, the time series is labelled looking only at the last timestep.
        If `all`, all timepoints in the timeseries are labelled.

    Returns
    -------
    X : array of shape `(N, n_points, n_features)`
        Numpy array, where:
        - `N` is the number of timeseries
        - `n_points` is the number of timesteps in each timeseries
        - `n_features` is the number of feature in the timeseries

    y : array of shape `(N,)` or `(N, n_points)`
        Numpy array of shape `(N,)` is `label_type=="last"`, or `(N, n_points)` if `label=="all"`.

    """

    start = 0
    end = df.shape[0]
    
    t0 = t1 = start
    X, y = [], []
    delta_list = []
    WRONG_CODE = 1 #Change Point Label
    while t1+window < end:
        delta_list.append(df.iloc[t0:t1+window, :].index) # -1 index excluded!
        t0 = t0 + stride
        t1 = t1 + stride 

    if label_type == "last":
        for chunk_delta in delta_list:
            df_index = df.index
            
            chunk_values = df.loc[df_index.intersection(chunk_delta)]
            ch_y = chunk_values.iloc[:, -1].values
            ch_x = chunk_values.iloc[:, :-1].values
            an = 1 if np.nonzero(ch_y == WRONG_CODE).sum()>0 else 0
            
            y.append(an)
            X.append(ch_x)
    elif label_type == "all":
        for chunk_delta in delta_list:
            df_index = df.index
            chunk_values = df.loc[df_index.intersection(chunk_delta)].iloc[:, -1]
            ch_y = chunk_values.iloc[:, -1].values
            ch_x = chunk_values.iloc[:, :-1].values
            an = np.nonzero(ch_y == WRONG_CODE)
            y.append(an)
            X.append(ch_x)
    
    X = np.stack([df for df in X])
    y = np.stack(y)
    
    return X, y

def load_pickle_dataset(path= '../../../dataset/5Year_Training.pkl', 
                        save_area= 823634):
    #Dataset.pkl for reinforcement learning
    with gzip.open(path, 'rb') as f:
        test = pickle.load(f)
    
    #Training Is for Unsupervised Learning(Becuz this pkl another wrong reputations code&data.)
    # with gzip.open('../../../dataset/5Year_Dataset.pkl', 'rb') as f:
    #     test = pickle.load(f)
    # print(test)
    
    #2Years for only 823634 Area
    AREA = save_area # 823634
    test[AREA].keys()
    
    SENSOR = ['SO2', 'PM10', 'O3', 'NO2',  
            'CO', 'NOX', 'NO','PM25', ]
    CODE = [f'{s}_CODE' for idx, s in enumerate(SENSOR)]

    LABEL = (test[AREA][CODE]==1).astype(int)
    test[AREA][SENSOR].head()
    #Anomaly Range 
    #test[AREA][SENSOR] = test[AREA][SENSOR].iloc[17544:35064, :] 

    for s,c in zip(SENSOR, CODE):
        os.makedirs(f'{AREA}', exist_ok=True)
        res = pd.concat([test[AREA][s], LABEL[c]], axis=1)
        res.to_csv(os.path.join(f'{AREA}', f'{c}.csv'), header=True, sep=',', index_label='Index')

if __name__ == "__main__":
    
    X, y = load_air_ds(path='../data/823634/CO_CODE.csv', 
                mode='train',
                window= "168H")
    