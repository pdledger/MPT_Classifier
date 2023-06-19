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
# Probflow for Bayesian approaches
import probflow as pf
# Use NLPClassifier for hyper parameter search before
# using Bayesian NN
from sklearn.neural_network import MLPClassifier

from skopt.space import Integer
from skopt.space import Real
from skopt.space import Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize

def Train_Probflow(Model,X_train_norm,Y_train,Number_Of_Classes):

    D=np.shape(X_train_norm)[1] # D is the number of features
    # Probflow requires float 32 data type
    x=X_train_norm.astype('float32')
    y=Y_train.astype('float32')

    if Model == "LogisticRegression_Bayesian":
        model = pf.LogisticRegression(D, k=Number_Of_Classes)
        model.fit(x, y, lr=0.01, epochs=200)
        #model.posterior_plot()

    if Model == "ProbFlowNN_Bayesian":
        model = pf.DenseClassifier([D, 32, 32, Number_Of_Classes])
        model.fit(x, y)
        
    if Model == "ProbFlowNN_Bayesian_Opt":
        print("Performing hyper parameter search")
        dim_num_dense_layers = Integer(name='num_dense_layers', low=1, high=3)
        dim_num_dense_nodes = Integer(name='num_dense_nodes', low=16, high=64)

        dimensions = [dim_num_dense_layers,
                     dim_num_dense_nodes]

        default_parameters = [1, 16]

##        def evaluate_model(p):
##            num_dense_layers, num_dense_nodes = p
##            list=[D]
##            for L in range(int(num_dense_layers)):
##                list.append(int(num_dense_nodes))
##            list.append(Number_Of_Classes)
##            model = pf.DenseClassifier(list)#[D, int(p[0]), int(p[0]), K])
##            model.fit(x, y, lr=0.01, epochs=200)
##            estimate=model.metric('mse',x,y)
##            # convert from a maximizing score to a minimizing score
##            print(list,estimate)
##            return 1.0 - estimate
# Idea - use a deterministic model to find the optmimum parmeters
        def evaluate_model(p):
            num_dense_layers, num_dense_nodes = p
            list=[num_dense_nodes]
            for L in range(int(num_dense_layers)-1):
                list.append(int(num_dense_nodes))
            model = MLPClassifier(hidden_layer_sizes=tuple(list),random_state=1, max_iter=300).fit(X_train_norm, Y_train)
            estimate=model.score(X_train_norm,Y_train)
            # convert from a maximizing score to a minimizing score
            print(list,estimate)
            return 1.0 - estimate      

        
        res = gp_minimize(evaluate_model,
                         dimensions=dimensions,
                         n_calls=30,
                         x0=default_parameters,
                         random_state=123)
        print("Found optimumum settings")
        print(res.x)
        print(res.fun)
        # Now train a Bayesian NN using the same hyper paramters.
        
        num_dense_layers, num_dense_nodes = res.x
        list=[D]
        for L in range(int(num_dense_layers)):
            list.append(int(num_dense_nodes))
        list.append(Number_Of_Classes)
        model = pf.DenseClassifier(list)#[D, int(p[0]), int(p[0]), K])
        model.fit(x, y, lr=0.01, epochs=200)



##
##      def model_builder(hp):
##            
##            # Tune the number of units in the first Dense layer
##            # Choose an optimal value between 32-512
##            hp_units = hp.Int('units', min_value=16, max_value=128, step=16)
##        
##            model = pf.DenseClassifier([D, hp_units, hp_units, Number_Of_Classes])
##            return model 
##
##        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
##        tuner.search(X_train_norm, Y_train, epochs=50, validation_split=0.2, callbacks=[stop_early])
##
##        # Get the optimal hyperparameters
##        best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
##        modelopt = tuner.hypermodel.build(best_hps)
##        model = tuner.hypermodel.build(best_hps)
##        model.fit(X_train_norm, Y_train)
        
    return model
