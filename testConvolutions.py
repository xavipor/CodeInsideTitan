#!/usr/bin/env python
import numpy as np
import theano
T = theano.tensor
floatX = theano.config.floatX
import theano.tensor.nnet.conv3d2d
 
# define inputs and filters
batchsize     = 1
in_channels   = 1
in_time       = 3
in_width      = 2
in_height     = 2
flt_channels  = 1
flt_time      = 3
flt_width     = 2
flt_height    = 2

input_shape = (batchsize, in_time, in_channels, in_height, in_width)
x = np.arange(np.prod(input_shape)).astype(np.float32)
x = np.reshape(x,input_shape)
inputs = theano.shared(x, borrow = True, name='inputs')

flt_shape = (flt_channels, flt_time, in_channels, flt_height, flt_width)
W = np.zeros(flt_shape,dtype=np.float32)
W[0,0,0,0,0] = 1 # When W[0,0,0,0,0] = 1, W[0,1,0,0,0] = 1 and W[0,2,0,0,0] = 1, I always get the same conv3d2d.conv3d outputs. So there must be something wrong. 
filters = theano.shared(W, borrow=True, name='filters')
bias = theano.shared(np.zeros(flt_channels,np.float32),name='bias')

# define first variant (conv3d2d)
convA = T.nnet.conv3d2d.conv3d(
	signals=inputs,  # Ns, Ts, C, Hs, Ws
	filters=filters, # Nf, Tf, C, Hf, Wf
	signals_shape=(batchsize, in_time, in_channels, in_height, in_width),
	filters_shape=(flt_channels, flt_time, in_channels, flt_height, flt_width),
	border_mode='valid')
convA = convA + bias.dimshuffle('x','x',0,'x','x')

"""
# define second variant (conv3D), should be the same
filters_flip = filters[:,::-1,:,::-1,::-1]  # flip time, width and height
convB = T.nnet.conv3D(
	V=inputs.dimshuffle(0,3,4,1,2),  # (batch, row, column, time, in channel)
	W=filters_flip.dimshuffle(0,3,4,1,2), # (out_channel, row, column, time, in channel)
	b=bias,
	d=(1,1,1))
convB = convB.dimshuffle(0,3,4,1,2)  # (batchsize, time, channels, height, width)

"""
# compile both
print "Compiling..."
funcA = theano.function([], convA)
#funcB = theano.function([], convB)

# run both
print "Executing..."
A = funcA()
#B = funcB()

# compare
print "A.shape =", A.shape
#print "B.shape =", B.shape
print "A = ", A
#print "B = ", B
#print "allclose(A, B) =", np.allclose(A, B)
