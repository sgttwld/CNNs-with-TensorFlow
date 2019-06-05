"""
Simple CNN from scratch for MNIST classification using core tensorflow 
Author: Sebastian Gottwald
Project: https://github.com/sgttwld/classification
Date: 2019-05-19
"""

import numpy as np
import tensorflow as tf
import math, os, sys, time
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

## custom progress bar
def print_progress(i, tot, acc, acc_str, bar_length=30, wait=False):
	filled_length = int(round(bar_length * i / tot))
	bar = '|' * filled_length + '-' * (bar_length - filled_length)
	sys.stdout.write('\r%s/%s |%s| %s %s' % (i, tot, bar, acc_str+':', acc)),
	if i == tot-1 and not(wait):
		sys.stdout.write('\n')
	sys.stdout.flush()

## import MNIST data and normalize
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train/255, x_test/255


class StrideOperator(object):

	def __init__(self,dimIn,dimCut,dimStride=(1,1)):
		self.dimIn = dimIn				# input size
		self.dimCut = dimCut 			# cut size
		self.dimStride = dimStride 		# stride sizes
		self.dimOut = [math.floor((self.dimIn[i]-self.dimCut[i])/self.dimStride[i])+1 for i in [0,1]]
		self.inds = self.get_inds_stride()	# collection of slices for the stride

	def get_inds_slice(self,begin):
		"""
		Returns the indices for a single slice of the input, which in case of convolution 
		will be multiplied by the convolution filter, the sum of which will be one entry 
		in the convolution.
		'begin' can have shape (...,n,m), where ... will be prefixed to the indices) 
		"""
		inds=[]
		for i in range(begin[-2],begin[-2]+self.dimCut[0]):
			for j in range(begin[-1], begin[-1]+self.dimCut[1]):
				inds.append([*begin[:-2],i,j])
		return np.reshape(inds,(self.dimCut[0],self.dimCut[1],len(begin))).tolist()
		
	def get_inds_stride(self,num=1):
		"""
		Collects the indices returned by the get_inds(begin) method for 
		each slice (different `begin`s of the slice) through the input.
		'num' determines the shape of the indices: 
			- num=1: indices are of the form [i,j]
			- num>1: indices are of the form [k,i,j] where k=0,...,num-1 
		"""
		if num == 1:
			indices = []
			for i in range(0,self.dimOut[0]):
				for j in range(0,self.dimOut[1]):
					begin = [i*self.dimStride[0],j*self.dimStride[1]]
					indices.append(self.get_inds_slice(begin))
		elif num > 1:
			indices = []
			for k in range(num):
				indices_num = []
				if num == 1:
					prefix=[]
				elif num > 1:
					prefix = [k]	
				for i in range(0,self.dimOut[0]):
					for j in range(0,self.dimOut[1]):
						begin = [*prefix,i*self.dimStride[0],j*self.dimStride[1]]
						indices_num.append(self.get_inds_slice(begin))
				indices.append(indices_num)
		return tf.constant(indices)


## convolution operation
class Convolution(StrideOperator):
	"""
	Creates a convolution operation, where the dimensions of input and filter are 
	specified during initialization.
	"""

	def __init__(self,dimIn,dimCut,numF,dimStride):
		StrideOperator.__init__(self,dimIn,dimCut,dimStride)
		self.numF = numF
		self.dimOut = (numF, *self.dimOut)		# shape: (numF,cutOut_x,cutOut_y)

	def conv(self,A,F):
		"""
		Performs the convolution of A with filter F, i.e. slices the input into 
		pieces of the shape of the filter F, applies F to each slice, and sums 
		up the entries of the result of the multiplication to obtain the entries 
		of the final result of the convolution
		"""
		A_slice = tf.gather_nd(params=A,indices=self.inds)	# shape: (#slices,cutA_x,cutA_y)
		r = tf.einsum('ijk,ljk->li',A_slice,F)				# shape: (numF,#slices)
		return tf.reshape(r,self.dimOut)					# shape: (numF,dimout_x,dimout_y)


## pooling operation
class Pooling(StrideOperator):
	"""
	Creates a pooling operation, where the dimensions of input and filter are 
	specified during initialization.
	"""
	def __init__(self,dimIn,numIn,dimCut,dimStride):
		StrideOperator.__init__(self,dimIn,dimCut,dimStride)
		self.numIn = numIn
		self.inds = self.get_inds_stride(numIn)		# shape: (numIn,#slices,dimCut[0],dimCut[1],len(index))
		self.dimOut = (numIn, *self.dimOut)			# shape: (numIn,poolOut_x,poolOut_y)

	def aver(self,A):
		"""
		Performs average pooling of A, i.e. slices the input into pieces, and returns
		the averages over the entries of each piece
		"""
		A_slice = tf.gather_nd(params=A,indices=self.inds)		# shape: (numIn,#slices,cutA_x,cutA_y)
		r = tf.reduce_mean(A_slice,axis=(2,3))					# shape: (numIn,#slices)
		return tf.reshape(r,self.dimOut)						# shape: (numIn,poolOut_x,poolOut_y)

	def max(self,A):
		"""
		Performs max pooling, i.e. slices the input into pieces, and returns
		the maximum of the entries of each piece
		"""
		A_slice = tf.gather_nd(params=A,indices=self.inds)		# shape: (numIn,#slices,cutA_x,cutA_y)
		r = tf.reduce_max(A_slice,axis=(2,3))					# shape: (numIn,#slices)
		return tf.reshape(r,self.dimOut)						# shape: (numIn,poolOut_x,poolOut_y)


## algorithm paramters
lr = .001		# learning rate
bs = 32			# batch size
numEp = 30		# number of episodes

## model parameters
dimIN = (28,28)			# size of input images
dimOUT = 10				# number of output nodes
dimF, numF = (5,5), 4 	# number and dimensions of convolutional filters

## initialize convolution and pooling operations
Conv = Convolution(dimIn=dimIN,dimCut=dimF,numF=numF,dimStride=(1,1))
Pool = Pooling(dimIn=(24,24),numIn=numF,dimCut=(2,2),dimStride=(2,2))

## resulting out-dims
dimC = Conv.dimOut 		 	# = (numF,24,24), output of convolutional layer
dimP = Pool.dimOut			# = (numF,12,12), output of pooling layer
dimFLAT = np.prod(dimP) 	# = numF*12*12, number of entries in the flattened layer

## weights and biases
F = tf.get_variable('conv_weights', (numF,*dimF), tf.float32, tf.glorot_uniform_initializer())
W = tf.get_variable('dense_weights',(dimFLAT,dimOUT), tf.float32)
b = tf.get_variable('dense_bias', (dimOUT), tf.float32, tf.zeros_initializer())

## placeholders for data
X = tf.placeholder(tf.float32,[None,dimIN[0],dimIN[1]])
Y = tf.placeholder(tf.int64,[None])
Y_1hot = tf.one_hot(Y,dimOUT,dtype=tf.float32)

## convolutional layer
C = tf.nn.relu(tf.map_fn(lambda A : Conv.conv(A,F), X))

## average pooling layer
P = tf.map_fn(lambda A : Pool.aver(A), C)

## flatten output of previous layers
FLAT = tf.layers.Flatten()(P)

## raw predictions
logits = tf.add(tf.matmul(FLAT,W),b)

## model output
p = tf.nn.softmax(logits)

# ## objective/loss
obj = -tf.reduce_mean(Y_1hot*tf.log(p)) 	# cross entropy

## classification error (for evaluation)
percent_corr = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(p,axis=1),Y),tf.float32))
err = 1 - percent_corr

## optimizer
optimizer = tf.contrib.optimizer_v2.AdamOptimizer(learning_rate=lr,beta1=.9, beta2=.999,epsilon=1e-08)
# optimizer = tf.train.AdamOptimizer(learning_rate=lr,beta1=.9, beta2=.999,epsilon=1e-08,name='Adam')
train_op = optimizer.minimize(obj)

## initializer
init = tf.global_variables_initializer()

## running the TF session
with tf.Session() as sess:

	## initializing
	sess.run(init)

	for n in range(0,numEp):
		numBatches = math.floor(len(x_train)/bs)
		t0, acc = time.time(), 0
		
		print('Ep:',n)
		for batch in range(0,numBatches):
			batch_X, batch_Y = x_train[batch*bs:(batch+1)*bs], y_train[batch*bs:(batch+1)*bs]
			sess.run(train_op,feed_dict={X: batch_X, Y: batch_Y})
			acc = (batch * acc + percent_corr.eval(session=sess,feed_dict={X:batch_X,Y:batch_Y}))/(batch+1)
			print_progress(batch, numBatches, round(acc,5), acc_str='acc', wait=True)
		T = round(time.time()-t0,2)
		acc_test = percent_corr.eval(session=sess,feed_dict={X:x_test,Y:y_test})
		sys.stdout.write(' time: %s test-acc: %s (error: %s%%)\n' % 
						(T, round(acc_test,3), round((1-acc_test)*100,3)))
