import numpy as np

#train
"""
wells = [1,2,3,4,5,6]
n_patches = [37,32,27,11,16,89]
n_sections = [18,36,18,18,18,18]

index = np.empty((4392,5),dtype=np.int16)

i=0
for w in range(len(wells)):
    for p in range(n_patches[w]):
        for s in range(n_sections[w]):
            index[i,0] = wells[w]
            
            if s==0:
                index[i,1] = n_sections[w]-1
            else:
                index[i,1] = s-1
            
            index[i,2] = s
            
            if s==n_sections[w]-1:
                index[i,3] = 0
            else:
                index[i,3] = s+1
                
            index[i,4] = p
            i+=1

print(i)
np.save('X_train/_new_index.npy',index)
"""

#test
wells = [7,8,9,10,11]
n_patches = [5,5,8,8,23]
n_sections = [36,18,18,18,18]

index = np.empty((972,5),dtype=np.int16)

i=0
for w in range(len(wells)):
    for p in range(n_patches[w]):
        for s in range(n_sections[w]):
            index[i,0] = wells[w]
            
            if s==0:
                index[i,1] = n_sections[w]-1
            else:
                index[i,1] = s-1
            
            index[i,2] = s
            
            if s==n_sections[w]-1:
                index[i,3] = 0
            else:
                index[i,3] = s+1
                
            index[i,4] = p
            i+=1

print(i)
np.save('X_test/_new_index.npy',index)