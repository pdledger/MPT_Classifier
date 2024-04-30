
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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
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
# Probflow for Bayesian approaches
import probflow as pf
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
    #DataSet_Name = 'Britishcoinstest_Al_0.84_Sig_2.4'
    #DataSet_Name = 'British_Coins/Coins_100_Al_0.84_Sig_2.4'
    #DataSet_Name = 'British-Coins/Coins_500_Al_0.84_Sig_2.4_Al_0.84_Sig_2.4'
    #DataSet_Name = 'British-Coins/Coins_1000_Al_0.84_Sig_2.4_Al_0.84_Sig_2.4'
    #DataSet_Name = 'British-Coins-Latest/Coins-500_Al_0.84_Sig_2.4'
    #DataSet_Name = 'British-Coins-Latest/Coins-1000_Al_0.84_Sig_2.4'
    DataSet_Name = 'British-Coins-Updated-1p-2p/Coins-1000_Al_0.84_Sig_12.5'
    #DataSet_Name = 'US_Coin_PL/Coins-200_Al_0.84_Sig_12.5'
    #DataSet_Name = 'US-Coins-PL-10Aug23/Coins-1000_Al_0.84_Sig_12.5'

    # Option to load external testing data from disk. Requires that external_file_loader.py be run first.
    Load_External_Data = True #False
    # Option to plot comparison figures between the input array of simulated data and the external test data.
    # Currently only supported for a single class test set.
    # This means that in the external file loader we need to select to read only one of the different coins say
    Plot_Comparison_Figures = False#True
    # Option to additionally save to disk: the model for each bootstrap iteration, the normalisation coefficients for each,
    # bootstrap iteration, and the input array for each model, Used for debugging.
    Full_Save = False

    # Option to use SVD to reduce the number of features
    Reduce_Features = True#False#True

    #Model to be used
    # Optional models
    # From Scikit-Learn:
    #'LogisticRegression', 'SVM', 'DecisionTree',
    # 'RandomForest', 'GradientBoost', 'MLP','MLP,(n1,n2,...,nn)'
    # From Tensor Flow
    # 'TensorflowNN', 'TensorflowNN_Opt'
    # From Probflow
    # 'LogisticRegression_Bayesian'


    # Models_to_run = ['LogisticRegression','SVM', 'DecisionTree', 'RandomForest', 'GradientBoost', 'MLP']
    Models_to_run = ["LogisticRegression_Bayesian","ProbFlowNN_Bayesian_Opt"]#'ProbFlowNN_Bayesian']#'LogisticRegression_Bayesian']
    #(string) 'LogisticRegression', 'SVM', 'SVM-RBF', 'DecisionTree', 'RandomForest',
    #'AdaBoost', 'GradientBoost', 'MLP' model to use when training the data



    #Features
    Features = ['Pri1', 'Pri2', 'Pri3']
    #Features = ['Eig1', 'Eig2', 'Eig3']
    #(list) list of features to be used options:
    #'Eig1','Eig2','Eig3','Pri1','Pri2','Pri3','Dev2','Dev3','Com'
    #Eigenvalues, Principal invarients, Deviatoric invarients, Comutator


    #How many times would you like to train the model
    Bootstrap_Repetitions = 1
    #(int) how many times to train the model to obtain an average accuracy


    #Noise
    if overwrite_noise_level is False:
        Noise_Levels = [5,10,20,40]#,10,15,20,40]#, 10, 15, 25]
    else:
        Noise_Level = SNR_array
    # Noise_Levels = [False]
    #(list) of (False or float or string)
    #False if no noise to be added
    #float in dB 20=> 10%, 40=> 1% actual noise
    #string '75% x' for % of instances to add x noise to


    #Define the colours to use
    PYCOL=['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf']

    # Define the Probabalistic classifiers (non-Bayesian) remove 'LogisticRegression_Bayesian'
    Probabalistic_Classifiers = ['LogisticRegression','RandomForest','AdaBoost','GradientBoost','MLP','TensorflowNN', 'TensorflowNN_Opt']
    # Define fully Bayesian classifiers
    Bayesian_Classifiers =['LogisticRegression_Bayesian','ProbFlowNN_Bayesian','ProbFlowNN_Bayesian_Opt']
    # Define Sckit Learn classifiers
    Scikit_Classifiers = ['LogisticRegression', 'SVM', 'DecisionTree', 'RandomForest', 'GradientBoost', 'MLP']
    # Define Tensor flow classifiers
    Tenflow_Classifiers = ['TensorflowNN', 'TensorflowNN_Opt']
    # Define Probflow classifiers
    Probflow_Classifiers = ['LogisticRegression_Bayesian','ProbFlowNN_Bayesian','ProbFlowNN_Bayesian_Opt']

    
    for model in Models_to_run:
        if model in Bayesian_Classifiers:
            Bootstrap_Repetitions = 1 # Do not perform bootstrap as we already
        # get output as a probability distribution.

    Main_loop(Noise_Levels,DataSet_Name,Load_External_Data,Plot_Comparison_Figures,
              Full_Save,Models_to_run,Features,Bootstrap_Repetitions,PYCOL,
              Probabalistic_Classifiers,Bayesian_Classifiers,Scikit_Classifiers,
              Tenflow_Classifiers,Probflow_Classifiers,Reduce_Features)





    ########################################################################


    #main script







if __name__ == '__main__':
    main()