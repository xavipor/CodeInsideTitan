import numpy as np
import scipy.io as sio
import sys
import cPickle,h5py
import tensorflow as tf


def createPlaceHolders(n_H,n_W,n_C,n_D):
    """
    Create placeholder for the session
    
    Arguments:
        n_H -- Height of the input image 
        n_W -- Width of the input image
        n_D -- Depth of the input image
        n_C -- Channels of the input image
        n_y -- number of classes
    """

    X = tf.placeholder(tf.float32,[None,n_D,n_H,n_W,n_C],name="X")

    return X,Y

def initializeVariables(parameters):

	"""
		Just to load the weights that I previously saved as numpy arrays, from the flatten version (Fully connected layers) of the convolutional net
		that was "fine tunned" with the patches from the hospital patients. 
		AFter all the wights are loaded, the variables for tensorflow are created. 
	"""
	path ='/home/jdominguezmartinez/pruebas/Microbleeds/cmb-3dcnn-code-v1.0/demo/code/lib/SavedModels/WeightsTrained/'
	W_L0 = np.load(path+'W0.npy')
	b_L0 = np.load(path+'b0.npy')

	W_L1 = np.load(path+'W1.npy')
	b_L1 = np.load(path+'b.npy')

	W_L2 = np.load(path+'W2.npy')
	b_L2 = np.load(path+'b2.npy')

	W_L3 = np.load(path+'W3.npy')
	W_L3c = np.reshape(W_L3,(2,2,2,64,150))
	b_L3 = np.load(path+'b3.npy')

	W_L4 = np.load(path+'W4.npy')
	W_L4c = np.reshape(W_L4,(1,1,1,150,2))
	b_L4 = np.load(path+'b4.npy')

	#Load the weights into the tensorflow variables
    W0 = tf.Variable(W_L0, name="W0")
    W1 = tf.Variable(W_L1, name="W1")
    W2 = tf.Variable(W_L2, name="W2")
    W3 = tf.Variable(W_L3c, name="W3")
    W4 = tf.Variable(W_L4c, name="W4")

    b0 = tf.Variable(b_L0, name ="b0")
    b1 = tf.Variable(b_L1, name ="b1")
    b2 = tf.Variable(b_L2, name ="b2")
    b3 = tf.Variable(b_L3, name ="b3")
    b4 = tf.Variable(b_L4, name ="b4")      	

    parameters={"W0":W0,"b0":b0,"W1":W1,"b1":b1,"W2":W2,"b2":b2,"W3":W3,"b3":b3,"W4":W4,"b4":b4}

    return parameters


def forward_propagation(X,parameters):
    #This could have been done better defining a Customized Convolution Layer and the same for flatten layer. 
    #I mean, the result is the same but it would be more structured. 

    W_L0 = parameters["W0"]
    b_L0 = parameters["b0"]

    W_L1 = parameters["W1"]
    b_L1 = parameters["b1"]

    W_L2 = parameters["W2"]
    b_L2 = parameters["b2"]

    W_L3 = parameters["W3"]
    b_L3 = parameters["b3"]

    W_L4 = parameters["W4"]
    b_L4 = parameters["b4"]


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
    #P3 = tf.contrib.layers.flatten(A3)

    # RELU
    #Z4 = tf.add(tf.matmul(P3,W_L3),b_L3)
    Z4 =tf.nn.conv3d(A3,W_L3,strides=[1,1,1,1,1],padding='VALID') 
    Z4 = tf.nn.bias_add(Z4,b_L3)
    A4 = tf.nn.relu(Z4)

    Z5=tf.nn.conv3d(A4,W_L4,strides=[1,1,1,1,1],padding='VALID') 
    Z5 = tf.nn.bias_add(Z5,b_L4)
    #Z5 =tf.add(tf.matmul(A4,W_L4),b_L4)

    return Z5

def testWrapper(input_sizes,output_sizes,patch_size,clip_rate,M_layer,layer_num,maxpool_sizes,activations,dropout_rates,
                para_path,save_score_map_path,whole_volume_path,mode):
    files = os.listdir(whole_volume_path)
    n_cases = len(files)
    print 'Have {} cases to process'.format(n_cases)               
           
    in_height, in_width, in_time = input_sizes

    for case_counter in range(n_cases):
        print 'Processing case # {} ... '.format(case_counter + 1)
        # cut the whole volume into smaller blocks, otherwise GPU will be out of memory
        dim0_score_start_pos = []
        dim0_score_end_pos = []
        dim0_start_pos = []
        dim0_end_pos = []  
        for part in range(clip_rate[0]):
            dim0_score_start_pos.append(1+part*output_sizes[0]/clip_rate[0])
            dim0_score_end_pos.append((part+1)*output_sizes[0]/clip_rate[0])
            dim0_start_pos.append(2*M_layer*(1+part*output_sizes[0]/clip_rate[0]-1)+1)
            dim0_end_pos.append(2*M_layer*((part+1)*output_sizes[0]/clip_rate[0]-1)+patch_size[0])   
        dim0_pos = zip(dim0_start_pos,dim0_end_pos)
        dim0_score_pos = zip(dim0_score_start_pos,dim0_score_end_pos)

        dim1_score_start_pos = []
        dim1_score_end_pos = []
        dim1_start_pos = []
        dim1_end_pos = []
        for part in range(clip_rate[1]):
            dim1_score_start_pos.append(1+part*output_sizes[1]/clip_rate[1])
            dim1_score_end_pos.append((part+1)*output_sizes[1]/clip_rate[1])
            dim1_start_pos.append(2*M_layer*(1+part*output_sizes[1]/clip_rate[1]-1)+1)
            dim1_end_pos.append(2*M_layer*((part+1)*output_sizes[1]/clip_rate[1]-1)+patch_size[1])   
        dim1_pos = zip(dim1_start_pos,dim1_end_pos)
        dim1_score_pos = zip(dim1_score_start_pos,dim1_score_end_pos)

        dim2_score_start_pos = []
        dim2_score_end_pos = []
        dim2_start_pos = []
        dim2_end_pos = []
        for part in range(clip_rate[2]):
            dim2_score_start_pos.append(1+part*output_sizes[2]/clip_rate[2])
            dim2_score_end_pos.append((part+1)*output_sizes[2]/clip_rate[2])
            dim2_start_pos.append(2*M_layer*(1+part*output_sizes[2]/clip_rate[2]-1)+1)
            dim2_end_pos.append(2*M_layer*((part+1)*output_sizes[2]/clip_rate[2]-1)+patch_size[2])   
        dim2_pos = zip(dim2_start_pos,dim2_end_pos)
        dim2_score_pos = zip(dim2_score_start_pos,dim2_score_end_pos)

        score_mask = np.zeros((2,output_sizes[0],output_sizes[1],output_sizes[2]))

        data_path = whole_volume_path + str(case_counter+1) + '_' + mode + '.mat'
        data_set = np.transpose(np.array(h5py.File(data_path)['data']))      
        data_set = data_set - np.mean(data_set)
        data_set = data_set.reshape((data_set.shape[0],in_time,1,in_height,in_width))

        for dim2 in range (clip_rate[2]):
        	for dim1 in range(clip_rate[1]):
        		for dim0 in range(clip_rate[0]):
        			smaller_data = data_set[:,dim2_pos[dim2][0]-1:dim2_pos[dim2][1],:,dim1_pos[dim1][0]-1:dim1_pos[dim1][1],dim0_pos[dim0][0]-1:dim0_pos[dim0][1]]
        			
        			#Aqui va el run de la red neuronal y lo guardamos en smaller_score
        			with tf.Session() as sess:
        			    pdb.set_trace()
        			    Z5 = forward_propagation(X, W0,b0,W1,b1,W2,b2,W3,b3,W4,b4)
        			    init = tf.initialize_all_variables()
        			    sess.run(init)
        			    smaller_score = sess.run(Z5,{X:image})
        			    print("_____________________________The value of the convolutional output is... _________________")
        			    
        			    print(smaller_score.shape)
        			    pdb.set_trace()
        			    sess.close()

        			score_mask[:,dim0_score_pos[dim0][0]-1:dim0_score_pos[dim0][1],dim1_score_pos[dim1][0]-1:dim1_score_pos[dim1][1],dim2_score_pos[dim2][0]-1:dim2_score_pos[dim2][1]] = smaller_score

        #When we went through all the clips that we done from the big image....
        result_file_name = save_score_map_path + str(case_counter+1) + '_score_mask.mat'
        sio.savemat(result_file_name,{'score_mask':score_mask})
        print 'The score_mask saved path:', result_file_name
        print 'Case {} wrap over!'.format(case_counter+1)

def test_model():
	para_path=''
	whole_volume_path = '/home/jdominguezmartinez/pruebas/Microbleeds/cmb-3dcnn-code-v1.0/demo/result/mat_data/'
	result_path = '/home/jdominguezmartinez/pruebas/Microbleeds/cmb-3dcnn-code-v1.0/demo/result/'
	#All this parameters, are needed in order to reconstruct the score volume due to our crops
	patch_size = [16,16,10]
	input_sizes = [512,512,148]
	output_sizes = [249,249,70] #Calculated based on the reduction factor of the net based on the input size.
	clip_rate = [3,3,2] #How many times we crop the image in each dimension. We construct small blocks due to running out of memory
	layer_num = 5
	M_layer = 1
	maxpool_sizes = [(2,2,2),(1,1,1),(1,1,1),(1,1,1),(1,1,1)]
	activations = [relu,relu,relu,relu,None]
	dropout_rates = [0.2,0.3,0.3,0.3,0.3]
	save_score_map_path = result_path + 'score_map/'

	testWrapper(input_sizes,output_sizes,patch_size,clip_rate,M_layer,layer_num,maxpool_sizes,activations,dropout_rates,para_path,save_score_map_path,whole_volume_path,mode='test')

if __name__ == '__main__':
    try:
        test_model()
    except KeyboardInterrupt:
        sys.exit()
