import numpy as np
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


from new_dataset import patch_dataset
from trainer import train
from models import basic_FCN, UNet


file_path_X = 'X_train'
file_path_Y = 'Y_train'
index = np.load('X_train/_new_index.npy')

dataset = patch_dataset(file_path_X, file_path_Y,index,load_in_ram=False, numtype=torch.float32)

batch_size = 64

dataloader = DataLoader(dataset, batch_size, shuffle=True)

model = UNet(in_channels=3, out_channels=3, features=[32, 64, 128, 256])
model.load_state_dict(torch.load('models/pretrained_Unet.pth', weights_only=True))     
#model.to('mps')
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

loss_fn = criterion = torch.nn.CrossEntropyLoss()

for i in range(2):
    train(model, optimizer, loss_fn, dataloader, device='cpu', epochs=5)
    torch.save(model.state_dict(),'models/pretrained_Unet.pth')
