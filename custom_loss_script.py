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

model = UNet(in_channels=1, out_channels=3, features=[32, 64, 128, 256])
model.load_state_dict(torch.load('models/pretrained_Unet.pth', weights_only=True))
model.load_state_dict(torch.load('models/custom_loss_Unet.pth', weights_only=True))
model.eval()

file_path_X = 'X_test'
index = np.load('X_test/_index.npy')

dataset_test = patch_dataset_test(file_path_X,index,load_in_ram=False, numtype=torch.float32)

batch_size = 1

dataloader = DataLoader(dataset_test, batch_size, shuffle=True)

for x in tqdm(dataloader):
    #x = x.to('mps')
    #y = y.to('mps')
    logits = model(x)[0]
    pred = torch.argmax(logits, dim=0)
    
    pred_custom = custom_output(logits,10)
    

    pred = pred.cpu().detach().numpy()
    # Define RGB colors for each class
    colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]])  # Red, Green, Blue

    # Convert class indices to RGB colors
    segmentation_mask = colors[pred]  # Shape: (H, W, 3)
    segmentation_mask_custom = colors[pred_custom]

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].imshow(x[0].permute(1, 2, 0).cpu().detach().numpy())
    
    axs[0].set_title('Input image')

    axs[1].imshow(segmentation_mask)
    axs[1].set_title('Predicted Segmentation')
    
    axs[2].imshow(segmentation_mask_custom)
    axs[2].set_title('Predicted Segmentation custom')

    plt.show()