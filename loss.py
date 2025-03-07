import torch
import torch.nn as nn



class custom_loss(nn.Module):
    def __init__(self,lbda):
        super().__init__()
        self.lbda = lbda
        self.CELoss = nn.CrossEntropyLoss()
        self.L2Loss = nn.MSELoss()
    
    def forward(self,y_pred,y):

        avg_2_pred = y_pred[:,2,:,:]
        avg_1_pred = y_pred[:,1,:,:]
        

        avg_2_pred = torch.mean(avg_2_pred,dim=1)
        sig_2_pred = torch.std(avg_2_pred,dim=1)
        avg_2_pred = avg_2_pred*torch.arange(0,272,1).unsqueeze(0)/272
        mean_2_pred = torch.mean(avg_2_pred,dim=1)
        
        sig_1_pred = torch.std(avg_1_pred,dim=2)
        avg_1_pred = avg_1_pred*torch.arange(0,272,1).unsqueeze(0).unsqueeze(0)/272
        mean_1_pred = torch.mean(avg_1_pred,dim=2)
        
        avg_2 = y[:,2,:,:]
        avg_1 = y[:,1,:,:]
        
        avg_2[:,:,0] = 1
        avg_2[:,:,-1] = 1
        
        avg_2 = torch.mean(avg_2,dim=1)
        sig_2 = torch.sum(avg_2,dim=1)
        avg_2 = avg_2*torch.arange(0,272,1).unsqueeze(0)/272
        mean_2 = torch.mean(avg_2,dim=1)
        
        sig_1 = torch.sum(avg_1,dim=2)
        avg_1 = avg_1*torch.arange(0,272,1).unsqueeze(0).unsqueeze(0)/272
        mean_1 = torch.mean(avg_1,dim=2)
        
        
        loss_0 = self.CELoss(y_pred,y)
        loss_2 = self.L2Loss(mean_2_pred,mean_2) + self.L2Loss(sig_2_pred,sig_2)    
        loss_1 = (self.L2Loss(mean_1_pred,mean_1) + self.L2Loss(sig_1_pred,sig_1))/160
        
        return loss_0 + self.lbda*(loss_1+loss_2)
    
def custom_output(y_pred,sig_lim):
    avg_2_pred = y_pred[2,:,:]
    avg_1_pred = y_pred[1,:,:]

    avg_2_pred = torch.mean(avg_2_pred,dim=0)
    sig_2_pred = torch.std(avg_2_pred,dim=0)
    avg_2_pred = avg_2_pred*torch.arange(0,272,1).unsqueeze(0)
    mean_2_pred = torch.mean(avg_2_pred,dim=0)
    
    sig_1_pred = torch.std(avg_1_pred,dim=1)
    avg_1_pred = avg_1_pred*torch.arange(0,272,1).unsqueeze(0).unsqueeze(0)
    mean_1_pred = torch.mean(avg_1_pred,dim=1)
    
    output = torch.zeros((160,272),dtype=torch.int16)
    output[:,int(avg_2_pred-sig_2_pred/2):int(avg_2_pred+sig_2_pred/2)] = 2
    
    for i in range(160):
        if sig_1_pred[i] < sig_lim:
            output[i,int(avg_1_pred[i]-sig_1_pred[i]/2):int(avg_1_pred[i]+sig_1_pred[i]/2)] = 1
            
    return output
    
    
        