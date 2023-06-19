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
from sklearn import model_selection, svm, tree, ensemble
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


def Train_Scikit(Model,X_train_norm,Y_train):
    #            # Fit the model on training set
    #            if Model == 'LogisticRegression':
    #                model = LogisticRegression(max_iter=1000).fit(X_train_norm, Y_train)
    #            if Model == 'SVM':
    #                model = svm.SVC(kernel='rbf',decision_function_shape='ovo').fit(X_train_norm, Y_train)
    #            if Model == 'DecisionTree':
    #                model = tree.DecisionTreeClassifier(max_depth=100).fit(X_train_norm, Y_train)
    #            if Model == 'RandomForest':
    #                model = ensemble.RandomForestClassifier(max_depth=100,n_estimators=100).fit(X_train_norm, Y_train)
    #            if Model == 'GradientBoost':
    #                model = ensemble.GradientBoostingClassifier(n_estimators=100).fit(X_train_norm, Y_train)
    #            if 'MLP' in Model:
    #                if Model == 'MLP':
    #                    model = MLPClassifier(hidden_layer_sizes=(100,100,25),random_state=1, max_iter=1000).fit(X_train_norm, Y_train)
    #                else:
    #                    Layers = Model.replace('MLP,','')
    #                    exec('model = MLPClassifier(hidden_layer_sizes='+Layers+',random_state=1, max_iter=1000).fit(X_train_norm, Y_train)')
    # save the model to disk
    #joblib.dump(model, 'DataSets/'+Dataset+'/'+Model_Save_Name+'.sav')
    # Fit the model on training set Optmised values for the paper
    if Model == 'LogisticRegression':
        model = LogisticRegression(max_iter=1000).fit(X_train_norm, Y_train)
    if Model == 'SVM':
        #model = svm.SVC(kernel='rbf',decision_function_shape='ovo').fit(X_train_norm, Y_train)
        model = svm.SVC(kernel='rbf',decision_function_shape='ovo',C=10e6,gamma=1).fit(X_train_norm, Y_train)
    if Model == 'DecisionTree':
        #model = tree.DecisionTreeClassifier(max_depth=100).fit(X_train_norm, Y_train)
        model = tree.DecisionTreeClassifier(max_depth=100).fit(X_train_norm, Y_train)
    if Model == 'RandomForest':
        #model = ensemble.RandomForestClassifier(max_depth=100,n_estimators=100).fit(X_train_norm, Y_train)
        model = ensemble.RandomForestClassifier(max_depth=100,n_estimators=100).fit(X_train_norm, Y_train)
    if Model == 'GradientBoost':
        #model = ensemble.GradientBoostingClassifier(n_estimators=100).fit(X_train_norm, Y_train)
        model = ensemble.GradientBoostingClassifier(n_estimators=50,max_depth=2).fit(X_train_norm, Y_train)
    if 'MLP' in Model:
        if Model == 'MLP':
            #model = MLPClassifier(hidden_layer_sizes=(100,100,25),random_state=1, max_iter=1000).fit(X_train_norm, Y_train)
            model = MLPClassifier(hidden_layer_sizes=(50,50,50),random_state=1, max_iter=300).fit(X_train_norm, Y_train)
        else:
            Layers = Model.replace('MLP,','')
            exec('model = MLPClassifier(hidden_layer_sizes='+Layers+',random_state=1, max_iter=1000).fit(X_train_norm, Y_train)')
    return model
