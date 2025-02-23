import numpy as np
import matplotlib.pyplot as plt

index = np.load('Y_train/_index.npy')
corrupt_index = np.load('Y_train/_index_corr.npy')
list_incomplete_patches = []

count = 0
for i in index:
    y = np.load('Y_train/'+i)
    print(y.shape)
    missing_lines = 0
    for j in range(y.shape[0]):
        if 1 not in y[j,:]:
            missing_lines += 1
    if missing_lines>0 and (i not in corrupt_index):
        count += 1
        list_incomplete_patches.append(i)
        plt.imshow(y,cmap='rainbow')
        plt.title(missing_lines)
        plt.show()
        
print(count)
list_incomplete_patches = np.asarray(list_incomplete_patches)
np.save('X_train/_index_incomp.npy',list_incomplete_patches)