
########################################################################

#This code was written by Ben Wilson with the supervison of Paul Ledger
#at the ZCCE, Swansea University

# Edit: 28 Mar 2022 - James Elgy
# Added functionallity for loading in external test datasets.
# Added plots for comparison between features for simulated training data and external test data.
# Added option to save out additional information. Used for debugging.
# Added pandas and seaborn to imports.
#
# Edit: 29 Mar 2022 - James Elgy
# In Noise_Adder: changed Tensor to be preallocated. We notice that this significantly improves the accuracy of the
# trained results. See code snippet bellow.
# Tensor = np.zeros((len(Frequencies), 9), dtype=complex)
# Tensor[:,:] = Tensors[int(X_train[i,0]),:,:]
# ...

########################################################################
#Import
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
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score
from scipy.stats import sem, t
import pandas as pd
import seaborn as sns
import time
# TensorFlow and tf.keras
import tensorflow as tf
sys.path.insert(0,"Functions")
from Main_loop import *
from Mean_confidence_interval import *


sys.path.insert(0,"Functions")
from Noise_adder import *
from PreProcessors import *


def main(SNR_array=[]):
    if SNR_array != []:
        overwrite_noise_level = True
    else:
        overwrite_noise_level = False


    #User Inputs

    #DataSet_Name = 'UK_Coin_Experimental_noisytestdata_magnetic_coins/UK_Coin_Al_0.84_Sig_2.4'
    #DataSet_Name = 'PLTest/Test15febmgcoins_Al_0.84_Sig_2.4'
    DataSet_Name = 'British_Coins/Coins_100_Al_0.84_Sig_2.4'



    # Option to load external testing data from disk. Requires that external_file_loader.py be run first.
    Load_External_Data = False #True
    # Option to plot comparison figures between the input array of simulated data and the external test data.
    # Currently only supported for a single class test set.
    Plot_Comparison_Figures = False
    # Option to additionally save to disk: the model for each bootstrap iteration, the normalisation coefficients for each,
    # bootstrap iteration, and the input array for each model, Used for debugging.
    Full_Save = False

    #Model to be used
    #Optional models 'LogisticRegression', 'SVM', 'DecisionTree', 'RandomForest', 'GradientBoost', 'MLP','MLP,(n1,n2,...,nn)'

    # Models_to_run = ['LogisticRegression','SVM', 'DecisionTree', 'RandomForest', 'GradientBoost', 'MLP']
    Models_to_run = ['TensorflowNN']#['TensorflowNN']#'LogisticRegression', 'MLP', 'GradientBoost']
    #Model = 'GradientBoost'
    #(string) 'LogisticRegression', 'SVM', 'SVM-RBF', 'DecisionTree', 'RandomForest',
    #'AdaBoost', 'GradientBoost', 'MLP' model to use when training the data



    #Features
    Features = ['Pri1', 'Pri2', 'Pri3']
    # Features = ['Eig1', 'Eig2', 'Eig3']
    #(list) list of features to be used options:
    #'Eig1','Eig2','Eig3','Pri1','Pri2','Pri3','Dev2','Dev3','Com'
    #Eigenvalues, Principal invarients, Deviatoric invarients, Comutator


    #How many times would you like to train the model
    Bootstrap_Repetitions = 1
    #(int) how many times to train the model to obtain an average accuracy


    #Noise
    if overwrite_noise_level is False:
        Noise_Levels = [5,10,15,20,40]#, 10, 15, 25]
    else:
        Noise_Level = SNR_array
    # Noise_Levels = [False]
    #(list) of (False or float or string)
    #False if no noise to be added
    #float in dB 20=> 10%, 40=> 1% actual noise
    #string '75% x' for % of instances to add x noise to


    #Define the colours to use
    PYCOL=['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf']

    #Define the Probabalistic classifiers
    Probabalistic_Classifiers = ['LogisticRegression','RandomForest','AdaBoost','GradientBoost','MLP','TensorflowNN', 'TensorflowNN_Opt']


    Main_loop(Noise_Levels,DataSet_Name,Load_External_Data,Plot_Comparison_Figures,Full_Save,Models_to_run,Features,Bootstrap_Repetitions,PYCOL,Probabalistic_Classifiers)





    ########################################################################


    #main script







if __name__ == '__main__':
    main()
