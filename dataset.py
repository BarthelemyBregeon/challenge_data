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
            
            for i in self.index:
                self.X[i] = np.load(self.file_path_X+'/'+i)
                
                y = np.load(self.file_path_Y+'/'+i)
                self.Y[i,j,k,y[j,k]] = 1 # faire fonctionner un truc comme ça
                
    
    def __len__(self):
        return self.len
    
    def getitem(self,i):
        if self.load_in_ram :
            return self.X[i],self.Y[i]
        else :
            return np.load(self.file_path_X+'/'+i),np.load(self.file_path_Y+'/'+i)# one hot encoding needed
        