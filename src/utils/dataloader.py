import math
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from .hasc_helper import load_hasc_ds
from .usc_ds_helper import load_usc_ds
from .yahoo_ds_helper import load_yahoo_ds
from .air_ds_helper import load_air_ds
import torch

class Load_Dataset(Dataset):
    def __init__(self, path:str, 
                ds_name:str, 
                win:int, 
                bs:int, 
                classes:int= 1,
                mode:str= 'train',
                **kwargs)->None:
        """_summary_
        Args:
            path (str): Path to dataset saved Folder
            ds_name (str): Dataset Name
            win (int): Sliding Window
            bs (int): BatchSize
            classes (int, optional): Number of classes. Defaults to 1.
            mode (str, optional): train or test stage. If you got all of data, notate 'all' 
                                    Defaults to 'train'.

        Raises:
            ValueError: Undefined Dataset

        Returns:
            None
        """
        self.path = path
        self.win_sz = win
        self.batch_size = bs
        self.n_classes = classes 
        load_dataset = {'HASC':load_hasc_ds, 
                        "USC":load_usc_ds, 
                        "YAHOO":load_yahoo_ds,
                        'AIR':load_air_ds}
        
        try:
            self.X, self.y = load_dataset[ds_name](path, window=2*win, mode=mode)
            
        except:
            raise ValueError("Undefined Dataset.")
        self.X = self.X.reshape(self.X.shape[:2]) if len(self.X.shape) > 2 else self.X
        self.y = self.y.reshape(-1, 1)
        self.y = np.where(self.y > 0, 1, 0)
        
        print(f"Total X shape:{self.X.shape}, Total y Shape: {self.y.shape}")
        
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, 
                                                        train_size=0.8, 
                                                        shuffle=True, 
                                                        stratify=self.y, 
                                                        random_state=13130132)
        
        if mode == 'train':
            self.X = X_train
            self.y = y_train
            print(f"Train X, y shape= {self.X.shape}, {self.y.shape}")
        elif mode == 'test' or mode == 'val':
            self.X = X_test
            self.y = y_test
            print(f"Test/Val X, y shape= {self.X.shape}, {self.y.shape}")
        
    
    def __str__(self):
        return f"<Dataset(N={len(self)}, batch_size={self.batch_size}, num_batches={self.get_num_batches()})>"
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        X = self.X[idx]
        y = self.y[idx]
        return X, y
    
    def ts_samples(self, bs_X, bs_y):
        X1 = bs_X[:,:, 1:self.win_sz+1]
        X2 = bs_X[:,:, -self.win_sz:]
        Y = bs_y[:,0]
        return X1, X2, Y
    
    def generate_batches(self, 
                        shuffle=False, 
                        drop_last=True):
        dataloader = DataLoader(dataset=self, 
                                batch_size=self.batch_size, 
                                collate_fn=self.collate_fn, 
                                shuffle=shuffle, 
                                drop_last=drop_last)
        for (X, y) in dataloader:
            # Previous Data of Split Window Size, 
            # Future Data of of Split Window Size
            X1, X2, y = self.ts_samples(X, y)
            yield (X1, X2), y
            
    def get_num_batches(self):
        return math.ceil(len(self)/self.batch_size)

    def collate_fn(self, batch):
        """Processing on a batch."""
        # Get inputs
        X = torch.FloatTensor(np.array([entry[0] for entry in batch])).unsqueeze(1)
        if self.n_classes == 1:
            y = torch.FloatTensor(np.array([entry[1] for entry in batch]).astype(np.int32))
            # y = y.squeeze(1)
        else:
            y = torch.LongTensor(np.array([entry[1] for entry in batch]).astype(np.int32)).squeeze(1)
        return X, y
    
    def get_weight(self, verbose=True):
        # Class weights
        from collections import Counter
        try:
            class_cnt = dict(Counter(self.y))
            all_class = sum(class_cnt.values())
            
            class_weights = {key: 1 - (count/all_class)
                                for key, count in class_cnt.items()}
        except:
            raise("Classes num ratio/Weight Invalid!")
        
        if verbose:
            print (f"Class/Counts: {class_cnt},\nclass weights: {class_weights}")
        return class_weights
    