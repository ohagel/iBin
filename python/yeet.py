import numpy as np

labels = np.loadtxt('dataset/labels.txt', delimiter=",", dtype=np.int32)
labels[:,2] = labels[:,2]-1
np.savetxt('dataset/labels.txt', labels, delimiter=",", fmt="%d")