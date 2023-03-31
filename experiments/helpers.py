from tensorflow.keras.utils import to_categorical

def normalizing(X, y):
  y_one_hot = to_categorical(y, num_classes=3)
  X_normal = X/255
  return X_normal, y_one_hot