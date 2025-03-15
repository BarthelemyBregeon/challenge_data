from new_dataset import patch_dataset_test
import numpy as np
import torch
from models import UNet
import os

file_path_X = 'X_test'

index = np.load('X_test/_new_index.npy')

dataset_test = patch_dataset_test(file_path_X,index,load_in_ram=False, numtype=torch.float32)

model = UNet(in_channels=3, out_channels=3, features=[32, 64, 128, 256])
model.load_state_dict(torch.load('models/pretrained_Unet_15.pth', weights_only=True))


predictions = []
with torch.no_grad():
    for i in range(len(dataset_test)):
        x,filename = dataset_test[i]
        logits = model(x.unsqueeze(0))[0]
        pred = torch.argmax(logits,dim=0).detach().numpy()
        predictions.append(pred)
        
        # Save the prediction
        np.save(f'predictions/{filename}', pred)

predictions = np.array(predictions)
print(predictions.shape)  # Should print (num_images, 160, 272)
































