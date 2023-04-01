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



def define_model(input_shape=(256, 256, 5), num_classes=3):
    model = unet_2d(input_shape=input_shape, num_classes=num_classes)
    model.compile(optimizer='adam',
                  loss=categorical_crossentropy,
                  metrics=['accuracy'])
    return model

def train_model(model, x_train, y_train, x_val, y_val, epochs=100):
    early_stop = EarlyStopping(monitor='accuracy', patience=5)
    history = model.fit(x=x_train, y=y_train, epochs=epochs,
                        validation_data=(x_val, y_val),
                        callbacks=[early_stop])
    return history

def save_model(model, model_name, saving_path='../models'):
    model.save(f'{saving_path}/model_{model_name}.h5')

def save_model_history(history, model_name, saving_path='../models'):
    with open(f'{saving_path}/history_{model_name}.pkl', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)


def predict_model(model, x_train, x_val, x_test):
    pred_train = model.predict(x_train)
    pred_val = model.predict(x_val)
    pred_test = model.predict(x_test)
    return pred_train, pred_val, pred_test

def evaluate_predictions(x_train, x_val, x_test, y_train, y_val, y_test, pred_train, pred_val, pred_test, training_dates, validation_dates, testing_dates, tile_size, step_size, count):
    metrics_train = EvaluationMetrics(x_train, x_val, x_test, y_train, y_val, y_train, pred_train, training_dates, validation_dates, testing_dates, tile_size, step_size, count)
    metrics_val = EvaluationMetrics(x_train, x_val, x_test, y_train, y_val, y_val, pred_val, training_dates, validation_dates, testing_dates, tile_size, step_size, count)
    metrics_test = EvaluationMetrics(x_train, x_val, x_test, y_train, y_val, y_test, pred_test, training_dates, validation_dates, testing_dates, tile_size, step_size, count)
    return {'metrics_train': metrics_train,
            'metrics_val': metrics_val,
            'metrics_test': metrics_test}

def save_metrics(metrics, model_name, saving_path='../metrics'):
    with open(f'{saving_path}/metrics_test_{model_name}.pkl', 'wb') as file:
        pickle.dump(metrics['metrics_test'], file)
    with open(f'{saving_path}/metrics_val_{model_name}.pkl', 'wb') as file:
        pickle.dump(metrics['metrics_val'], file)
    with open(f'{saving_path}/metrics_train_{model_name}.pkl', 'wb') as file:
        pickle.dump(metrics['metrics_train'], file)

def get_mean_jaccard(all_metrics):
  jaccard_array = []
  for idx, metric in enumerate(all_metrics):
    print(metric.jaccard)
    jaccard_array.append(metric.jaccard)

  print()
  print(f'Mean jaccard index: {sum(jaccard_array)/10}')
  print()
  print(f'Worst index: {min(jaccard_array)}')
  print(f'Best index: {max(jaccard_array)}')
  print(f'Variance: {max(jaccard_array)-min(jaccard_array)}')