import tensorflow as tf
import numpy as np
import h5py
import cPickle


path='/home/jdominguezmartinez/pruebas/Microbleeds/cmb-3dcnn-code-v1.0/demo/code/lib/PesosPruebaTf/'
W_L0 = np.load(path+'W_L2.npy')


# Add ops to save and restore all the variables.
saver = tf.train.import_meta_graph("/home/jdominguezmartinez/pruebas/Microbleeds/cmb-3dcnn-code-v1.0/demo/code/lib/SavedModels/my_test_model-30.meta")

# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
with tf.Session() as sess:
  # Restore variables from disk.
  saver.restore(sess,("/home/jdominguezmartinez/pruebas/Microbleeds/cmb-3dcnn-code-v1.0/demo/code/lib/SavedModels/my_test_model-30"))
  print("Model restored.")
  # Check the values of the variables
  v1 = sess.graph.get_tensor_by_name("W4:0")
  v1_1 = v1.eval()
    
  sess.close()

print("____________________________________________________")
print(v1_1)



path='/home/jdominguezmartinez/pruebas/Microbleeds/cmb-3dcnn-code-v1.0/demo/code/lib/PesosPruebaTf/'
W_L0 = np.load(path+'W_L2.npy')


# Add ops to save and restore all the variables.
saver = tf.train.import_meta_graph("/home/jdominguezmartinez/pruebas/Microbleeds/cmb-3dcnn-code-v1.0/demo/code/lib/SavedModels/my_test_model-40.meta")
# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
with tf.Session() as sess:
  # Restore variables from disk.
  saver.restore(sess,("/home/jdominguezmartinez/pruebas/Microbleeds/cmb-3dcnn-code-v1.0/demo/code/lib/SavedModels/my_test_model-40"))
  print("Model restored.")
  # Check the values of the variables
  v2 = sess.graph.get_tensor_by_name("W4:0")
  v2_1 = v2.eval()
  sess.close()

print("____________________________________________________")
print(v2_1-v1_1)
