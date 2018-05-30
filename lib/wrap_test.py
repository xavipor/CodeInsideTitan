import numpy as np
import theano.tensor as T
import theano
import time,os
import theano.tensor.nnet.conv3d2d
import scipy.io as sio
import sys
import pdb

import cPickle,h5py
from lib.max_pool import max_pool_3d
from lib.relu import relu
from lib.load_mat import load_mat,sharedata
floatX = theano.config.floatX

class ConvPoolLayer(object):
    def __init__(self, input, filter, base, activation, poolsize, dtype = theano.config.floatX):
        
        """
        Allocate a Conv3dLayer with shared variable internal parameters.
      
        :type input: theano.tensor
        :param input: 5D matrix -- (batch_size, time, in_channels, height, width)
        
        :type filter: 
        :param filter: 5D matrix -- (num_of_filters, flt_time, in_channels, flt_height, flt_width)
        
        :type filters_shape: tuple or list of length 5
        :param filter_shape:(number_of_filters, flt_time,in_channels,flt_height,flt_width)
        
        :type base: tuple or list of length number_of_filters
        :param base:(number_of_filters)
        
        :param activation: non-linear activation function, typically relu or tanh 
        
        :poolsize: tuple or list of length 3
        :param poolsize: the pooling stride, typically (2,2,2)              
        """
        
        self.input = input
        self.W = filter
        self.b = base
        
        # do the 3d convolution --- have flip
        conv_out = theano.tensor.nnet.conv3d2d.conv3d(
            signals = self.input,   
            filters = self.W,         
            signals_shape = None,
            filters_shape = None,
            border_mode = 'valid')  # the convolution stride is 1
        
        conv = conv_out + self.b.dimshuffle('x','x',0,'x','x')
        
        if poolsize is None:
            pooled_out = conv
        else:
            pooled_out = max_pool_3d(input=conv, ds=poolsize, ignore_border=True)
        
        # non-linear function
        self.output = ( 
            pooled_out if activation is None 
            else activation(pooled_out)
        )
        
       # store parameters of this layer
        self.params = [self.W, self.b]
        
class LogisticRegression(object):
    def __init__(self,input,x,y,z):
        # flatten the input feature volumes into vectors
        self.input = input.reshape((z,2,y,x)).dimshuffle(0,2,3,1).reshape((x*y*z,2))
        # employ softmax to get prediction probabilities
        self.p_y_given_x = T.nnet.softmax(self.input)
        # reshape back into score volume
        self.score_map = self.p_y_given_x.reshape((z,y,x,2)).dimshuffle(3,2,1,0)  # dimension is z y x
        self.y_pred = T.argmax(self.p_y_given_x,axis=1)
    
class wrap_3dfcn(object):
    def __init__(self, input, layer_num, maxpool_sizes, activations, dropout_rates,
                para_path, final_size, show_param_label=False):
        """
        This is to efficiently wrap the whole volume with 3D FCN
        
        :type input: theano.tensor
        :param input: 5D matrix -- (batch_size, time, in_channels, height, width)
        
        :type layer_num: int
        :param layer_num: number of layers in the network
        
        :type maxpool_sizes: list
        :param maxpool_sizes: maxpooling sizes of each layer
        
        :param activations: non-linear activation function, typically relu or tanh
        
        :type dropout_rates: list
        :param dropout_rates: dropout rate of each layer
        
        :param para_path: saved model paththeano
        
        :type final_size: list of length 3
        :param final_size: output score volume size -- (final_time, final_height, final_width) 
        """
	 
        f = open(para_path,'r') 
        params = cPickle.load(f) 
        if show_param_label:
            print 'params loaded!' 
                      
        self.layers = []
        next_layer_input = input
        for layer_counter in range(layer_num):
            W = params[layer_counter*2]
            pdb.set_trace()
            
            b = params[layer_counter*2+1]
            if show_param_label:
                print 'layer number:{0}, size of filter and base: {1} {2}'.format(layer_counter, W.shape.eval(), b.shape.eval())
            
            next_layer = ConvPoolLayer(
                    input = next_layer_input,
                    filter = W*(1-dropout_rates[layer_counter]),
                    base = b,
                    activation = activations[layer_counter],
                    poolsize = maxpool_sizes[layer_counter])
            
            self.layers.append(next_layer)
            next_layer_input = next_layer.output
            layer_counter += 1
        
        final_time, final_height, final_width = final_size
        score_volume_layer = LogisticRegression(
                input = self.layers[-1].output,
                x = final_width,
                y = final_height,
                z = final_time)
                    
        self.score_volume = score_volume_layer.score_map            
                    
                    
def test_wrapper(input_sizes,output_sizes,patch_size,clip_rate,M_layer,layer_num,maxpool_sizes,activations,dropout_rates,
                para_path,save_score_map_path,whole_volume_path,mode):    
                           
    files = os.listdir(whole_volume_path)
    n_cases = len(files)
    print 'Have {} cases to process'.format(n_cases)               

    
    start_time = time.time()
    for case_counter in xrange(n_cases):
        pdb.set_trace()
        data_path = whole_volume_path + str(74) + '.mat'
        data_set = np.transpose(np.array(h5py.File(data_path)['patchFlatten']))   
        data_set = data_set - np.mean(data_set)
        data_set = data_set.reshape((data_set.shape[0],10,1,16,16))
        wrapper = wrap_3dfcn(input = theano.shared(np.asarray(data_set,theano.config.floatX),borrow = True),
            layer_num = layer_num,
            maxpool_sizes = maxpool_sizes,
            activations = activations,
            dropout_rates = dropout_rates,
            para_path = para_path,
            final_size = (dim2_score_end_pos[0], dim0_score_end_pos[0], dim1_score_end_pos[0]))    
        test_model = theano.function(inputs = [], outputs = wrapper.score_volume)
        smaller_score = test_model()
        #smaller_score=wrapper.score_volume
        score_mask[:,dim0_score_pos[dim0][0]-1:dim0_score_pos[dim0][1],dim1_score_pos[dim1][0]-1:dim1_score_pos[dim1][1],dim2_score_pos[dim2][0]-1:dim2_score_pos[dim2][1]] = smaller_score
        # score_mask[:,dim0_score_pos[dim0][0]-1:dim0_score_pos[dim0][1],dim1_score_pos[dim1][0]-1:dim1_score_pos[dim1][1],dim2_score_pos[dim2][0]-1:dim2_score_pos[dim2][1]] = smaller_score.eval()
                    
        result_file_name = save_score_map_path + str(case_counter+1) + '_score_mask.mat'
        print 'The score_mask saved path:', result_file_name
        sio.savemat(result_file_name,{'score_mask':score_mask})
        print 'Case {} wrap over!'.format(case_counter+1)
    end_time = time.time()
    print 'time spent {} seconds.'.format((end_time-start_time))
                      
