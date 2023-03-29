import numpy as np
import tensorflow as tf

def normalizing(X, y):
  print(format('y shape: {}', y.shape))
  y_one_hot =  np.array([tf.one_hot(item, depth=3).numpy() for item in y])
  print(format('y one hot shape: {}', y_one_hot.shape))
  X_normal = X/255
  return X_normal, y_one_hot