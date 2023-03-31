from evaluation.evaluation_metrics import EvaluationMetrics
from models.unet_model import unet_2d
from keras.losses import categorical_crossentropy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import pickle

def normalizing(X, y):
  print('y shape: ', y.shape)
  print('Normalizing data...')
  y_one_hot = to_categorical(y, num_classes=3)
  print('y one hot shape: ', y_one_hot.shape)
  X_normal = X/255
  return X_normal, y_one_hot



def execute_training(count, x_train, x_val, x_test, y_train, y_val, y_test, training_dates, validation_dates, testing_dates, tile_size, step_size, saving_path):
  print(f'Start training number {count}')
  model = unet_2d(input_shape=(256, 256, 5), num_classes=3)
  model.compile(optimizer='adam',
                loss=categorical_crossentropy,
                metrics=['accuracy']) # are weights resetted?

  early_stop = EarlyStopping(monitor='accuracy', patience=5) 

  model_history = model.fit(x=x_train, y=y_train, epochs=100, validation_data=(x_val, y_val), callbacks=[early_stop]) 
  print('training completed')
  
  # saving model
  model_name = f'{tile_size}_{step_size}_run_{count}'
  model.save(f'../models/{saving_path}/model_{model_name}.h5')
  print('saving model completed')

  # saving model history
  with open(f'../models/{saving_path}/history_{model_name}.pkl', 'wb') as file_pi:
      pickle.dump(model_history.history, file_pi)
  print('saving history completed')

  # making predictions
  pred_test = model.predict(x_test)
  pred_val = model.predict(x_val)
  pred_train = model.predict(x_train)
  print('making predictions completed')
  print(pred_test.shape)
  print(pred_val.shape)
  print(pred_train.shape)

  # calculating metrics
  metrics_test = EvaluationMetrics(x_train, x_val, x_test, y_train, y_val, y_test, pred_test, training_dates, validation_dates, testing_dates, tile_size, step_size, count)
  metrics_val = EvaluationMetrics(x_train, x_val, x_test, y_train, y_val, y_val, pred_val, training_dates, validation_dates, testing_dates, tile_size, step_size, count)
  metrics_train = EvaluationMetrics(x_train, x_val, x_test, y_train, y_val, y_train, pred_train, training_dates, validation_dates, testing_dates, tile_size, step_size, count)
  print('calculating metrics completed')

  # saving metrics
  with open(f'../metrics/{saving_path}/metrics_test{model_name}.pkl', 'wb') as file_pi:
      pickle.dump(metrics_test, file_pi)
  with open(f'../metrics/{saving_path}/metrics_val{model_name}.pkl', 'wb') as file_pi:
      pickle.dump(metrics_val, file_pi)
  with open(f'../metrics/{saving_path}/metrics_train{model_name}.pkl', 'wb') as file_pi:
      pickle.dump(metrics_train, file_pi)
  print('saving metrics completed')

  return  {'metrics_test': metrics_test, 'metrics_val': metrics_val, 'metrics_train': metrics_train}


def random_print():
   print('random print')