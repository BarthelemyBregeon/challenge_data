import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class patch_dataset(Dataset):
    
    def __init__(self, file_path_X, file_path_Y, index, load_in_ram = False):
        self.index = index
        self.file_path_X = file_path_X
        self.file_path_Y = file_path_Y
        self.len = index.shape[0]
        self.load_in_ram = load_in_ram
        
        if self.load_in_ram :
            self.X = torch.empty((self.len,160,272),dtype=torch.float64)
            self.Y = torch.zeros((self.len,160,272,3),dtype=torch.float64)
            
            for i, file in zip(range(self.len),self.index):
                self.X[i] = torch.from_numpy(np.load(self.file_path_X+'/'+file))
               
                y = torch.from_numpy(np.load(self.file_path_Y+'/'+file))
                self.Y[i] = torch.nn.functional.one_hot(y.to(torch.int64), num_classes=3)
                
    
    def __len__(self):
        return self.len
    
    def __getitem__(self,i):
        if self.load_in_ram :
            return self.X[i],self.Y[i]
        else :
            x = torch.from_numpy(np.load(self.file_path_X+'/'+index[i]))
            y = torch.from_numpy(np.load(self.file_path_Y+'/'+index[i]))
            y = torch.nn.functional.one_hot(y.to(torch.int64), num_classes=3)
            return x,y
        
file_path_X = 'X_train'
file_path_Y = 'Y_train'
index = np.load('X_train/_index_good.npy')

dataset = patch_dataset(file_path_X, file_path_Y,index,load_in_ram=False)
        