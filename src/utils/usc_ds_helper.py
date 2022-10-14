from math import floor
import numpy as np
import scipy.io as sio
import os
from sklearn.model_selection import train_test_split

def ts_samples(mbatch, win):
    x = mbatch[:,1:win+1]
    y = mbatch[:,-win:]
    lbl = mbatch[:,0]
    return x, y, lbl


def load_usc_ds(path, window, mode='train'):

    X, lbl = extract_windows(path, window, mode)
    
    
    if mode == "all":
        return X, lbl
    X_train, X_test, y_train, y_test = train_test_split(X, lbl, 
                                                        train_size=0.8, 
                                                        shuffle=True, 
                                                        stratify=lbl, 
                                                        random_state=13130132)
    # train_size = int(floor(0.8 * X.shape[0]))
    if mode == "train":
        # trainx = X[0:train_size]
        # trainlbl = lbl[0:train_size]
        # idx = np.arange(trainx.shape[0])
        # np.random.shuffle(idx)
        # trainx = trainx[idx,]
        # trainlbl = trainlbl[idx]
        print('train samples : ', y_train.shape)
        return X_train, y_train

    else:
        # testx = X[train_size:]
        # testlbl = lbl[train_size:]
        print(f'Test shape {X_test.shape} and number of change points {len(np.where(y_test > 0)[0])}')
        return X_test, y_test


def extract_windows(path, window_size, mode=None):
    windows = []
    lbl = []
    dataset = sio.loadmat(os.path.join(path, "usc.mat"))

    ts = np.array(dataset['Y'])
    ts = ts[:,0] #Time Series
    
    cp = np.array(dataset['L'])
    cp = cp[:,0] #Change Point Label
    
    num_cp = 0
    for i in range(0, ts.shape[0] - window_size, 5):
        windows.append(np.array(ts[i:i + window_size]))
        #print("TS",ts[i:i+window_size])
        is_cp = np.where(cp[i:i + window_size] == 1)[0]
        if is_cp.size == 0:
            is_cp = [0]
        else:
            num_cp += 1
        lbl.append(is_cp[0])

    print(f"number of samples : {len(windows)}"
            f" number of samples with change point : {num_cp}")
    windows = np.array(windows)

    return windows, np.array(lbl)