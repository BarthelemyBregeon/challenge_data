import numpy as np
import torch
from torch.utils.data import Dataset

class patch_dataset(Dataset):
    
    def __init__(self, file_path_X, file_path_Y, index, numtype=torch.float64, load_in_ram = False):
        self.index = np.char.mod('%d', index)
        self.file_path_X = file_path_X
        self.file_path_Y = file_path_Y
        self.len = self.index.shape[0]
        self.load_in_ram = load_in_ram
        self.numtype = numtype
        
        if self.load_in_ram :
            self.X = torch.empty((self.len,160,272,3),dtype=self.numtype)
            self.Y = torch.zeros((self.len,160,272,3),dtype=self.numtype)
            
            for i, file in zip(range(self.len),self.index):
                x1 = torch.from_numpy(np.load(self.file_path_X+'/well_'+file[0]+'_section_'+file[1]+'_patch_'+file[4]+'.npy'))
                x2 = torch.from_numpy(np.load(self.file_path_X+'/well_'+file[0]+'_section_'+file[2]+'_patch_'+file[4]+'.npy'))
                x3 = torch.from_numpy(np.load(self.file_path_X+'/well_'+file[0]+'_section_'+file[3]+'_patch_'+file[4]+'.npy'))
                if x1.shape[1] == 160:
                    x1 = torch.nn.functional.pad(x1, (0,272-160), value=0)
                    x2 = torch.nn.functional.pad(x2, (0,272-160), value=0)
                    x3 = torch.nn.functional.pad(x3, (0,272-160), value=0)
                self.X[i,:,:,0] = x1
                self.X[i,:,:,1] = x2
                self.X[i,:,:,2] = x3
               
                y = torch.from_numpy(np.load(self.file_path_Y+'/well_'+file[0]+'_section_'+file[2]+'_patch_'+file[4]+'.npy'))
                if y.shape[1] == 160:
                    y = torch.nn.functional.pad(y, (0,272-160), value=0)
                self.Y[i] = torch.nn.functional.one_hot(y.to(torch.int64), num_classes=3)
                
    
    def __len__(self):
        return self.len
    
    def __getitem__(self,i):
        if self.load_in_ram :
            return self.X[i].permute(2,0,1),self.Y[i].permute(2, 0, 1)
        else :
            x1 = torch.from_numpy(np.load(self.file_path_X+'/well_'+self.index[i,0]+'_section_'+self.index[i,1]+'_patch_'+self.index[i,4]+'.npy'))
            x2 = torch.from_numpy(np.load(self.file_path_X+'/well_'+self.index[i,0]+'_section_'+self.index[i,2]+'_patch_'+self.index[i,4]+'.npy'))
            x3 = torch.from_numpy(np.load(self.file_path_X+'/well_'+self.index[i,0]+'_section_'+self.index[i,3]+'_patch_'+self.index[i,4]+'.npy'))
            if x1.shape[1] == 160:
                x1 = torch.nn.functional.pad(x1, (0,272-160), value=0)
                x2 = torch.nn.functional.pad(x2, (0,272-160), value=0)
                x3 = torch.nn.functional.pad(x3, (0,272-160), value=0)
            
            x = torch.cat((x1.unsqueeze(0),x2.unsqueeze(0),x3.unsqueeze(0)),dim=0)
            y = torch.from_numpy(np.load(self.file_path_Y+'/well_'+self.index[i,0]+'_section_'+self.index[i,2]+'_patch_'+self.index[i,4]+'.npy'))
            if y.shape[1] == 160:
                y = torch.nn.functional.pad(y, (0,272-160), value=0)
            y = torch.nn.functional.one_hot(y.to(torch.int64), num_classes=3)
            
            return x.to(self.numtype),y.permute(2, 0, 1).to(self.numtype)
    
        
    
class patch_dataset_test(Dataset):
    
    def __init__(self, file_path_X, index, numtype=torch.float64, load_in_ram = False):
        self.index = np.char.mod('%d', index)
        self.file_path_X = file_path_X
        self.len = index.shape[0]
        self.load_in_ram = load_in_ram
        self.numtype = numtype
        
        if self.load_in_ram :
            self.X = torch.empty((self.len,160,272,3),dtype=self.numtype)
            
            for i, file in zip(range(self.len),self.index):
                x1 = torch.from_numpy(np.load(self.file_path_X+'/well_'+file[0]+'_section_'+file[1]+'_patch_'+file[4]+'.npy'))
                x2 = torch.from_numpy(np.load(self.file_path_X+'/well_'+file[0]+'_section_'+file[2]+'_patch_'+file[4]+'.npy'))
                x3 = torch.from_numpy(np.load(self.file_path_X+'/well_'+file[0]+'_section_'+file[3]+'_patch_'+file[4]+'.npy'))
                if x1.shape[1] == 160:
                    x1 = torch.nn.functional.pad(x1, (0,272-160), value=0)
                    x2 = torch.nn.functional.pad(x2, (0,272-160), value=0)
                    x3 = torch.nn.functional.pad(x3, (0,272-160), value=0)
                self.X[i,:,:,0] = x1
                self.X[i,:,:,1] = x2
                self.X[i,:,:,2] = x3
               
    
    def __len__(self):
        return self.len
    
    def __getitem__(self,i):
        if self.load_in_ram :
            return self.X[i].permute(2,0,1),self.index[i]
        else :
            x1 = torch.from_numpy(np.load(self.file_path_X+'/well_'+self.index[i,0]+'_section_'+self.index[i,1]+'_patch_'+self.index[i,4]+'.npy'))
            x2 = torch.from_numpy(np.load(self.file_path_X+'/well_'+self.index[i,0]+'_section_'+self.index[i,2]+'_patch_'+self.index[i,4]+'.npy'))
            x3 = torch.from_numpy(np.load(self.file_path_X+'/well_'+self.index[i,0]+'_section_'+self.index[i,3]+'_patch_'+self.index[i,4]+'.npy'))
            if x1.shape[1] == 160:
                x1 = torch.nn.functional.pad(x1, (0,272-160), value=0)
                x2 = torch.nn.functional.pad(x2, (0,272-160), value=0)
                x3 = torch.nn.functional.pad(x3, (0,272-160), value=0)
            
            x = torch.cat((x1.unsqueeze(0),x2.unsqueeze(0),x3.unsqueeze(0)),dim=0)

            return x.to(self.numtype),'well_'+self.index[i,0]+'_section_'+self.index[i,2]+'_patch_'+self.index[i,4]+'.npy'
        
        