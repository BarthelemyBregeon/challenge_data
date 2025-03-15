import numpy as np
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


from dataset import patch_dataset
from dataset import patch_dataset_test
from trainer import train
from models import basic_FCN, UNet
from loss import custom_loss, custom_output

file_path_X = 'X_train'
file_path_Y = 'Y_train'
index = np.load('X_train/_index.npy')

dataset = patch_dataset(file_path_X, file_path_Y,index,load_in_ram=True, numtype=torch.float32)

batch_size = 64

dataloader = DataLoader(dataset, batch_size, shuffle=True)

model = UNet(in_channels=1, out_channels=3, features=[32, 64, 128, 256])
#model.to('mps')
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#loss_fn = criterion = torch.nn.CrossEntropyLoss()

#for i in range(10):
#    train(model, optimizer, loss_fn, dataloader, device='cpu', epochs=5)
#    torch.save(model.state_dict(),'models/pretrained_Unet.pth')

model.load_state_dict(torch.load('models/pretrained_Unet.pth', weights_only=True))

loss_fn = criterion = custom_loss(1e-2)
train(model, optimizer, loss_fn, dataloader, device='cpu', epochs=5)
torch.save(model.state_dict(),'models/custom_loss_Unet.pth')
