import numpy as np
import matplotlib.pyplot as plt

index = np.load('Y_train/_index.npy')
list_corrupted_patches = []

count = 0
for i in index:
    y = np.load('Y_train/'+i)
    if y[-1,0] ==-1:
        count += 1
        list_corrupted_patches.append(i)
       
print(count)
list_corrupted_patches = np.asarray(list_corrupted_patches)
np.save('X_train/_index_corr.npy',list_corrupted_patches)
np.save('Y_train/_index_corr.npy',list_corrupted_patches)