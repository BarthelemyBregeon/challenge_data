import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


Y = pd.read_csv('Y_train.csv', sep=',', header=None, chunksize=1)

i = 0
for chunk in Y:
    if i != 0:
        if i%10 == 0:
            print(i)
        y = chunk.to_numpy()
        y = np.squeeze(y,0)
        file_path = y[0]
        if y[1:].shape[0] == 25600:
            y = np.reshape(y[1:],[160,160]).astype(np.int16)
            np.save('Y_train/'+file_path,y)
        else:
            y = np.reshape(y[1:],[160,272]).astype(np.int16)
            np.save('Y_train/'+file_path,y)
    i+=1
    
    
"""Y = pd.read_csv('submission_csv_file_random_example.csv', sep=',', header=None, chunksize=1)

i = 0
index = []

for chunk in Y:
    if i != 0:
        if i%10 == 0:
            print(i)
        y = chunk.to_numpy()
        y = np.squeeze(y,0)
        file_path = y[0]
        index.append(file_path+'.npy')
        
    i+=1

index = np.asarray(index)
print(index)
np.save('X_test/_index.npy',index)     """


