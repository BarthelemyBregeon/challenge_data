import torch
import torch.nn as nn

import matplotlib.pyplot as plt



class custom_loss(nn.Module):
    def __init__(self,lbda):
        super().__init__()
        self.lbda = lbda
        self.CELoss = nn.CrossEntropyLoss()
        self.L2Loss = nn.MSELoss()
    
    def forward(self,y_pred,y):
        
        eps = 1e-5
        
        y_pred = nn.functional.softmax(y_pred,dim=1)

        avg_2_pred = y_pred[:,2,:,:]
        avg_1_pred = y_pred[:,1,:,:]
        
        avg_2_pred = torch.mean(avg_2_pred,dim=1)
        wid_2_pred = torch.sum(avg_2_pred,dim=1)
        avg_2_pred = (avg_2_pred*torch.arange(0,272,1).unsqueeze(0))/torch.sum(avg_2_pred,dim=1).unsqueeze(1)
        mean_2_pred = torch.sum(avg_2_pred,dim=1) 
        
        wid_1_pred = torch.sum(avg_1_pred,dim=2)
        avg_1_pred = (avg_1_pred*torch.arange(0,272,1).unsqueeze(0).unsqueeze(0))/torch.sum(avg_1_pred,dim=2).unsqueeze(2)
        mean_1_pred = torch.sum(avg_1_pred,dim=2)
        
        avg_2 = y[:,2,:,:]
        avg_1 = y[:,1,:,:]
        
        avg_2 = torch.mean(avg_2,dim=1)
        wid_2 = torch.sum(avg_2,dim=1)
        avg_2 = (avg_2*torch.arange(0,272,1).unsqueeze(0))/torch.sum(avg_2,dim=1).unsqueeze(1)
        mean_2 = torch.sum(avg_2,dim=1)
        
        wid_1 = torch.sum(avg_1,dim=2)
        avg_1 = (avg_1*torch.arange(0,272,1).unsqueeze(0).unsqueeze(0))/(torch.sum(avg_1,dim=2).unsqueeze(2) + eps)
        mean_1 = torch.sum(avg_1,dim=2)
        
        
        #loss_0 = self.CELoss(y_pred,y)
        loss_2 = self.L2Loss(mean_2_pred,mean_2) + self.L2Loss(wid_2_pred,wid_2)    
        loss_1 = (self.L2Loss(mean_1_pred,mean_1) + self.L2Loss(wid_1_pred,wid_1))/160
        
        #return loss_0 + self.lbda*(loss_1+loss_2)
        return loss_1+loss_2
    
def custom_output(y_pred,wid_min):  
    y_pred = nn.functional.softmax(y_pred,dim=0)
    
    avg_2_pred = y_pred[2,:,:]
    avg_1_pred = y_pred[1,:,:]

    avg_2_pred = torch.mean(avg_2_pred,dim=0)
    wid_2_pred = torch.sum(avg_2_pred,dim=0)
    avg_2_pred = (avg_2_pred*torch.arange(0,272,1))/torch.sum(avg_2_pred)
    mean_2_pred = torch.sum(avg_2_pred,dim=0) 
    
    wid_1_pred = torch.sum(avg_1_pred,dim=1)
    avg_1_pred = (avg_1_pred*torch.arange(0,272,1).unsqueeze(0))/torch.sum(avg_1_pred,dim=1).unsqueeze(1)
    mean_1_pred = torch.sum(avg_1_pred,dim=1)
    
    
    output = torch.zeros((160,272),dtype=torch.int16)
    output[:,int(mean_2_pred-wid_2_pred/2):int(mean_2_pred+wid_2_pred/2)] = 2
    
    for i in range(160):
        if wid_1_pred[i] > wid_min:
            output[i,int(mean_1_pred[i]-wid_1_pred[i]/2):int(mean_1_pred[i]+wid_1_pred[i]/2)] = 1
            
    return output
    
    
        