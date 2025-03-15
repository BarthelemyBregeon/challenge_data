import numpy as np
import matplotlib.pyplot as plt


colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]])  # Red, Green, Blue


fig, axs = plt.subplots(3, 2, figsize=(15, 5))

axs[0,0].imshow(np.load("X_train/well_1_section_0_patch_0.npy"))

axs[0,0].set_title('Input image')

y = np.load("Y_train/well_1_section_0_patch_0.npy")


axs[0,1].imshow(y)
axs[0,1].set_title('Ground Truth')

axs[1,0].imshow(np.load("X_train/well_1_section_1_patch_0.npy"))

axs[1,0].set_title('Input image')

y = np.load("Y_train/well_1_section_1_patch_0.npy")


axs[1,1].imshow(y)
axs[1,1].set_title('Ground Truth')



axs[2,0].imshow(np.load("X_train/well_1_section_2_patch_0.npy"))

axs[2,0].set_title('Input image')

y = np.load("Y_train/well_1_section_2_patch_0.npy")


axs[2,1].imshow(y)
axs[2,1].set_title('Ground Truth')



plt.show()

