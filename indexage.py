from os import listdir
from os.path import isfile, join
import numpy as np
import re

onlyfiles = [f for f in listdir('X_Train') if isfile(join('X_Train', f))]
onlyfiles = np.asarray(onlyfiles)
sorting = np.empty([onlyfiles.shape[0]],dtype=np.int16)

for i in range(onlyfiles.shape[0]):
    id_ = re.findall(r'\d+', onlyfiles[i])
    id_ = int(id_[0])*1000 + int(id_[1])*100 + int(id_[2])
    sorting[i] = id_

onlyfiles = onlyfiles[np.argsort(sorting)]

np.save('X_Train/_index.npy',onlyfiles)
np.save('Y_Train/_index.npy',onlyfiles)