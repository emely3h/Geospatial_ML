import numpy as np
import tensorflow as tf

def normalizing(X, y):
  y_one_hot =  np.array([tf.one_hot(item, depth=3).numpy() for item in y])
  X_normal = X/255
  return X_normal, y_one_hot