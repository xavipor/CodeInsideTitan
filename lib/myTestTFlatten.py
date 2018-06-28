import math
import numpy as np
import h5py
import cPickle
import matplotlib.pyplot as plt
import tensorflow as tf
import pdb

path='/home/jdominguezmartinez/pruebas/Microbleeds/cmb-3dcnn-code-v1.0/demo/code/lib/PesosPruebaTf/'
W_L0 = np.load(path+'W_L0.npy')
b_L0 = np.load(path+'b_L0.npy')

W_L1 = np.load(path+'W_L1.npy')
b_L1 = np.load(path+'b_L1.npy')

W_L2 = np.load(path+'W_L2.npy')
b_L2 = np.load(path+'b_L2.npy')

W_L3 = np.load(path+'W_L3.npy')
b_L3 = np.load(path+'b_L3.npy')

W_L4 = np.load(path+'W_L4.npy')
b_L4 = np.load(path+'b_L4.npy')
pdb.set_trace()


whole_volume_path = '/home/jdominguezmartinez/pruebas/Microbleeds/cmb-3dcnn-code-v1.0/demo/code/patches/'
data_path = whole_volume_path + str(74) + '.mat'
data_set = np.transpose(np.array(h5py.File(data_path)['patchFlatten']))
image =  data_set.reshape((data_set.shape[0],10,16,16,1))

def forward_propagation(X, W_L0,b_L0,W_L1,b_L1,W_L2,b_L2,W_L3,b_L3,W_L4,b_L4):
    if 'a' == 'a':

    
    # Retrieve the parameters from the dictionary "parameters" 

    
    ### START CODE HERE ###
    # CONV2D: stride of 1, padding 'SAME
        Z1 =tf.nn.conv3d(X,W_L0,strides=[1,1,1,1,1],padding='VALID') 
        Z1 = tf.nn.bias_add(Z1,b_L0)
        Z1= tf.nn.max_pool3d(Z1,ksize=(1,2,2,2,1),strides=(1,2,2,2,1),padding='VALID')
        A1 = tf.nn.relu(Z1)
        # MAXPOOL: window 8x8, sride 8, padding 'SAME'
        Z2 =tf.nn.conv3d(A1,W_L1,strides=[1,1,1,1,1],padding='VALID') 
        Z2 = tf.nn.bias_add(Z2,b_L1)
        A2 = tf.nn.relu(Z2)
        # CONV2D: filters W2, stride 1, padding 'SAME'
        Z3 =tf.nn.conv3d(A2,W_L2,strides=[1,1,1,1,1],padding='VALID') 
        Z3 = tf.nn.bias_add(Z3,b_L2)
        A3 = tf.nn.relu(Z3)
        # RELU
	P2 = tf.contrib.layers.flatten(A3)
	print("EEEEEEEEEEEEEEEEEEEE",P2.get_shape().as_list())
        Z4 = tf.add(tf.matmul(P2,W3),b3)
        A4 = tf.nn.relu(Z4)
	print(A4.get_shape().as_list())

        Z5 =tf.add(tf.matmul(A4,W4),b4)


        return Z5

X = tf.placeholder(tf.float32,[1,10,16,16,1],name="X")
W0 = tf.Variable(W_L0, name="W0")
W1 = tf.Variable(W_L1, name="W1")
W2 = tf.Variable(W_L2, name="W2")
W3 = tf.Variable(W_L3, name="W3")
W3 = tf.reshape(W3,shape=[512,150])
W4 = tf.Variable(W_L4, name="W4")
W4 = tf.reshape(W4,shape=[150,2])
b0 = tf.Variable(b_L0, name ="b0")
b1 = tf.Variable(b_L1, name ="b1")
b2 = tf.Variable(b_L2, name ="b2")
b3 = tf.Variable(b_L3, name ="b3")
b4 = tf.Variable(b_L4, name ="b4")
print("shape of b",b4.get_shape().as_list())


with tf.Session() as sess:
    pdb.set_trace()
    Z5 = forward_propagation(X, W0,b0,W1,b1,W2,b2,W3,b3,W4,b4)
    init = tf.initialize_all_variables()
    sess.run(init)
    value = sess.run(Z5,{X:image})
    np.save(path+'outputFlatten.npy',value)
    print(value)
    sess.close()

