import os
import sys
from sys import exit
import math
import csv
import joblib
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt

def Train_Tenflow(Model,X_train_norm,Y_train):
    if Model == 'TensorflowNN':
        # Setup a simple tensor flow model. Just one hidden layer with 128 neurons
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(Number_Of_Classes)
        ])
        model.compile(optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
        model.fit(X_train_norm, Y_train, epochs=10)
        #test_loss, test_acc = model.evaluate(X_test_norm,  Y_test, verbose=2)
        probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    if Model == 'TensorflowNN_Opt':

        def model_builder(hp):
            model = keras.Sequential()
            # Tune the number of units in the first Dense layer
            # Choose an optimal value between 32-512
            hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
            model.add(keras.layers.Dense(units=hp_units, activation='relu'))
            model.add(keras.layers.Dense(units=hp_units, activation='relu'))
            model.add(keras.layers.Dense(Number_Of_Classes))

            # Tune the learning rate for the optimizer
            # Choose an optimal value from 0.01, 0.001, or 0.0001
            hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                          loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                          metrics=['accuracy'])

            return model
        tuner = kt.Hyperband(model_builder,
                             objective='val_accuracy',
                             max_epochs=10,
                             factor=3,
                             directory='my_dir',
                             project_name='intro_to_kt',
                             overwrite=True)
        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        tuner.search(X_train_norm, Y_train, epochs=50, validation_split=0.2, callbacks=[stop_early])

        # Get the optimal hyperparameters
        best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

        print(f"""
        The hyperparameter search is complete. The optimal number of units in the first densely-connected
        layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
        is {best_hps.get('learning_rate')}.
        """)

        # Build the model with the optimal hyperparameters and train it on the data for 50 epochs
        modelopt = tuner.hypermodel.build(best_hps)
        history = modelopt.fit(X_train_norm, Y_train, epochs=50, validation_split=0.2)

        val_acc_per_epoch = history.history['val_accuracy']
        best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
        print('Best epoch: %d' % (best_epoch,))

        model = tuner.hypermodel.build(best_hps)

        # Retrain the model using the hyper parameters and the best choice of epochs
        model.fit(X_train_norm, Y_train, epochs=best_epoch, validation_split=0.2)
        probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    return model, probability_model

