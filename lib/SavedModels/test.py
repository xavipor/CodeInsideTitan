
import tensorflow as tf
import numpy as np
import h5py
import cPickle


path='/home/jdominguezmartinez/pruebas/Microbleeds/cmb-3dcnn-code-v1.0/demo/code/lib/PesosPruebaTf/'
W_L0 = np.load(path+'W_L3.npy')

# Add ops to save and restore all the variables.
saver = tf.train.import_meta_graph("/home/jdominguezmartinez/pruebas/Microbleeds/cmb-3dcnn-code-v1.0/demo/code/lib/SavedModels/my_test_model-30.meta")

# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
with tf.Session() as sess:
  # Restore variables from disk.
  saver.restore(sess,("/home/jdominguezmartinez/pruebas/Microbleeds/cmb-3dcnn-code-v1.0/demo/code/lib/SavedModels/my_test_model-30"))
  print("Model restored.")
  # Check the values of the variables
  v1 = sess.graph.get_tensor_by_name("W3:0")
  v1_1 = v1.eval()
    
  sess.close()

print("____________________________________________________")
print(v1_1)



path='/home/jdominguezmartinez/pruebas/Microbleeds/cmb-3dcnn-code-v1.0/demo/code/lib/PesosPruebaTf/'



# Add ops to save and restore all the variables.
saver = tf.train.import_meta_graph("/home/jdominguezmartinez/pruebas/Microbleeds/cmb-3dcnn-code-v1.0/demo/code/lib/SavedModels/my_test_model-40.meta")
# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
with tf.Session() as sess:
  # Restore variables from disk.
  saver.restore(sess,("/home/jdominguezmartinez/pruebas/Microbleeds/cmb-3dcnn-code-v1.0/demo/code/lib/SavedModels/my_test_model-40"))
  print("Model restored.")
  # Check the values of the variables
  v2 = sess.graph.get_tensor_by_name("W3:0")
  v3 = sess.graph.get_tensor_by_name("W3a:0")

  v2_1 = v2.eval()
  v3_1 = v3.eval()
  v3_1_1 = np.reshape(v2_1,(2,2,2,64,150))
  sess.close()

print("____________________________________________________")

print(v3_1_1 - v3_1)

print("____________________________________________________")

print(W_L0 - v3_1)
