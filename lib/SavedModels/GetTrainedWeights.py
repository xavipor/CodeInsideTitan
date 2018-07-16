import tensorflow as tf
import numpy as np

pathToSave = '/home/jdominguezmartinez/pruebas/Microbleeds/cmb-3dcnn-code-v1.0/demo/code/lib/SavedModels/WeightsTrained/'

# Add ops to save and restore all the variables.
saver = tf.train.import_meta_graph("/home/jdominguezmartinez/pruebas/Microbleeds/cmb-3dcnn-code-v1.0/demo/code/lib/SavedModels/my_test_model_AD13_all_L2bigR_-1800.meta")

with tf.Session() as sess:
  # Restore variables from disk.
  saver.restore(sess,("/home/jdominguezmartinez/pruebas/Microbleeds/cmb-3dcnn-code-v1.0/demo/code/lib/SavedModels/my_test_model_AD13_all_L2bigR_-1800"))
  print("Model restored.")
  # Check the values of the variables and save them
  for i in range(5):
	Wt = sess.graph.get_tensor_by_name("W"+str(i)+":0")
	bt = sess.graph.get_tensor_by_name("b"+str(i)+":0")
	W = Wt.eval() #To get a numpy array from the tensor
	print(W.shape)
	b = bt.eval() #To get a numpy array from the tensor
	np.save(pathToSave+"W"+str(i),W)
	np.save(pathToSave+"b"+str(i),b)
  sess.close()

print("All weights and biases saved" )
