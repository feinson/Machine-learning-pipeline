import numpy as np

np.random.seed(2)
arr = np.random.randint(0,10,size=(9, 12))

print(arr)
print([len(np.unique(arr[i])) for i in range(len(arr))])