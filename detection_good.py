import numpy as np
import matplotlib.pyplot as plt

index = np.load('Y_train/_index.npy')
corrupt_index = np.load('X_train/_index_corr.npy')
incomplete_index = np.load('X_train/_index_incomp.npy')
list_good_patches = []

count = 0
for i in index:
    if (i not in incomplete_index) and (i not in corrupt_index):
        count += 1
        list_good_patches.append(i)
        
        
print(count)
list_good_patches = np.asarray(list_good_patches)
np.save('X_train/_index_good.npy',list_good_patches)
np.save('Y_train/_index_good.npy',list_good_patches)