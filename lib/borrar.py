
import numpy as np


path ='/home/jdominguezmartinez/pruebas/Microbleeds/cmb-3dcnn-code-v1.0/demo/code/lib/SavedModels/WeightsTrained/'
W_L0 = np.load(path+'W0.npy')
b_L0 = np.load(path+'b0.npy')

W_L1 = np.load(path+'W1.npy')
b_L1 = np.load(path+'b1.npy')

W_L2 = np.load(path+'W2.npy')
b_L2 = np.load(path+'b2.npy')

W_L3 = np.load(path+'W3.npy')
b_L3 = np.load(path+'b3.npy')

W_L4 = np.load(path+'W4.npy')
b_L4 = np.load(path+'b4.npy')


print(W_L3.shape)
print(W_L4.shape)

#a = np.reshape(W_L3, (2,2,2,150,2))
#print(a.shape)
