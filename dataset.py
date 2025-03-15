import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class patch_dataset(Dataset):
    
    def __init__(self, file_path_X, file_path_Y, index, numtype=torch.float64, load_in_ram = False):
        self.index = index
        self.file_path_X = file_path_X
        self.file_path_Y = file_path_Y
        self.len = index.shape[0]
        self.load_in_ram = load_in_ram
        
        if self.load_in_ram :
            self.X = torch.empty((self.len,160,272),dtype=numtype)
            self.Y = torch.zeros((self.len,160,272,3),dtype=numtype)
            
            for i, file in zip(range(self.len),self.index):
                x = torch.from_numpy(np.load(self.file_path_X+'/'+file))
                if x.shape[1] == 160:
                    x = torch.nn.functional.pad(x, (0,272-160), value=0)
                self.X[i] = x
               
                y = torch.from_numpy(np.load(self.file_path_Y+'/'+file))
                if y.shape[1] == 160:
                    y = torch.nn.functional.pad(y, (0,272-160), value=0)
                self.Y[i] = torch.nn.functional.one_hot(y.to(torch.int64), num_classes=3)
                
    
    def __len__(self):
        return self.len
    
    def __getitem__(self,i):
        if self.load_in_ram :
            return self.X[i].unsqueeze(0),self.Y[i].permute(2, 0, 1)
        else :
            x = torch.from_numpy(np.load(self.file_path_X+'/'+index[i]))
            if x.shape[1] == 160:
                x = torch.nn.functional.pad(x, (0,272-160), value=0)
                
            y = torch.from_numpy(np.load(self.file_path_Y+'/'+index[i]))
            if y.shape[1] == 160:
                y = torch.nn.functional.pad(y, (0,272-160), value=0)
            y = torch.nn.functional.one_hot(y.to(torch.int64), num_classes=3)
            
            return x.unsqueeze(0),y.permute(2, 0, 1)
    
        
    
class patch_dataset_test(Dataset):
    
    def __init__(self, file_path_X, index, numtype=torch.float64, load_in_ram = False):
        self.index = index
        self.file_path_X = file_path_X
        self.len = index.shape[0]
        self.load_in_ram = load_in_ram
        
        if self.load_in_ram :
            self.X = torch.empty((self.len,160,272),dtype=numtype)
            
            for i, file in zip(range(self.len),self.index):
                x = torch.from_numpy(np.load(self.file_path_X+'/'+file))
                if x.shape[1] == 160:
                    x = torch.nn.functional.pad(x, (0,272-160), value=0)
                self.X[i] = x
                
    
    def __len__(self):
        return self.len
    
    def __getitem__(self,i):
        if self.load_in_ram :
            return self.X[i].unsqueeze(0)
        else :
            x = torch.from_numpy(np.load(self.file_path_X+'/'+index[i]))
            return x.unsqueeze(0)
    

        
file_path_X = 'X_train'
file_path_Y = 'Y_train'
index = np.load('X_train/_index_good.npy')

dataset = patch_dataset(file_path_X, file_path_Y,index,load_in_ram=True)
        