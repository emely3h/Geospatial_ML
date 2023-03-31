from tensorflow.keras.utils import to_categorical

def normalizing(X, y):
  print('y shape: ', y.shape)
  print('Normalizing data...')
  y_one_hot = to_categorical(y, num_classes=3)
  print('y one hot shape: ', y_one_hot.shape)
  X_normal = X/255
  return X_normal, y_one_hot


if __name__ == '__main__':
  import numpy as np
  X = np.random.rand(10, 32, 32, 3)
  y = np.random.randint(0, 3, size=10)
  X_normal, y_one_hot = normalizing(X, y)