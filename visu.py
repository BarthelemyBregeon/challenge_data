import numpy as np
from matplotlib import pyplot as plt
import torch

index = np.load("Y_train/_index_corr.npy")
for i in index:

    colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]])  # Red, Green, Blue


    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    axs[0].imshow(np.load("X_train/"+i))
    
    axs[0].set_title('Input image')

    y = np.load("Y_train/"+i)

    
    axs[1].imshow(y)
    axs[1].set_title('Ground Truth')



    plt.show()