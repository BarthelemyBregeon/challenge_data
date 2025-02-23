import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


Y = pd.read_csv('Y_train.csv', sep=',', header=None, chunksize=1)
#Y = pd.read_csv('submission_csv_file_random_example.csv', sep=',', header=None, chunksize=1)

i = 0
for chunk in Y:
    if i != 0:
        if i%10 == 0:
            print(i)
        y = chunk.to_numpy()
        y = np.squeeze(y,0)
        file_path = y[0]
        y = np.reshape(y[1:],[160,272]).astype(np.int16)
        #x = np.load('X_Train/'+file_path+'.npy')
        #x = np.load('X_Test/'+file_path+'.npy')
        #plt.imshow(x)
        plt.imshow(y,cmap='rainbow')
        if y[-1,0] == -1:
            plt.title("Attention")
        plt.show()
    i+=1
        


