import tensorflow as tf
import numpy as np
from tensorflow import keras
from IRM_methods import *
import time
import os

# comment following line if requred amount of gpu is available
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def simulate_hetero_fl():
    start_time = time.time()
    n_e = 4

    # load data for evaluation
    X_test = np.load('RIS_Data/X2.npy')
    y_test = np.load('RIS_Data/y2.npy')

    client_batch = int(len(X_test) / (n_e - 1))
    y_pred_ = []
    for i in range(n_e - 1):
        if i != n_e - 2:
            x_test_client = X_test[i * client_batch: (i + 1) * client_batch]
            y_test_client = y_test[i * client_batch: (i + 1) * client_batch]
        else:
            x_test_client = X_test[i * client_batch:]
            y_test_client = y_test[i * client_batch:]
        rep_model = keras.models.load_model('models/rep_model')

        test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        z_test = rep_model(x_test_client)
        yts_ = tf.zeros_like(y_test_client, dtype=tf.float32)
        for e in range(0, n_e - 1):
            pred_model = keras.models.load_model('models/pred_model_env' + str(e + 1))
            yts_ = yts_ + pred_model(z_test)

        y_pred_.append(yts_)

    y_pred = tf.concat(y_pred_, axis=0)
    test_acc = float(test_accuracy(y_test, y_pred))
    elapsed_time = time.time() - start_time

    return elapsed_time


def simulate_fed_avg():
    start_time = time.time()
    n_e = 4

    X_test = np.load('RIS_Data/X2.npy')
    y_test = np.load('RIS_Data/y2.npy')

    client_batch = int(len(X_test) / (n_e - 1))
    y_pred_ = []
    for i in range(n_e - 1):
        if i != n_e - 2:
            x_test_client = X_test[i * client_batch: (i + 1) * client_batch]
        else:
            x_test_client = X_test[i * client_batch:]

        test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        model = keras.models.load_model('models/fed_avg')
        yts_ = model.predict(x_test_client)

        y_pred_.append(yts_)

    y_pred = tf.concat(y_pred_, axis=0)
    test_acc = float(test_accuracy(y_test, y_pred))
    elapsed_time = time.time() - start_time

    return elapsed_time


X_test = np.load('RIS_Data/X2.npy')
print('starting hetero fl simulation.')
fl_hetero_time = simulate_hetero_fl()
print('fl hetero simulation ended.')
print('Total time elapsed for fl hetero simulation ' + str(fl_hetero_time))
print('Average simulation time for one sample ' + str(fl_hetero_time / X_test.shape[0]))

print('--------------------------------')

X_test = np.load('RIS_Data/X2.npy')
print('starting FedAvg simulation.')
fed_avg_time = simulate_fed_avg()
print('FedAvg simulation ended.')
print('Total time elapsed for FedAvg simulation ' + str(fed_avg_time))
print('Average simulation time for one sample ' + str(fed_avg_time / X_test.shape[0]))
