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
#from sklearn import model_selection, svm, tree, ensemble
#from sklearn.linear_model import LogisticRegression
#from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score
from scipy.stats import sem, t
import pandas as pd
import seaborn as sns
import time
# TensorFlow and tf.keras
#import tensorflow as tf
#from tensorflow import keras
#import keras_tuner as kt
# Probflow for Bayesian approaches
#import probflow as pf
from Mean_confidence_interval import *
from Noise_adder import *
from PreProcessors import *
from Plotting import *
from Write_report import *
from My_confusion_matrix import *
from Bootstrap_UQ import *
from Bayesian_UQ import *

from External_data_save import *
from Train_Scikit import *
from Train_Tenflow import *
from Train_Probflow import *

from Output_snapshot import *

from Sklearn_results import *
from Tenflow_results import *
from Probflow_results import *
from Simple_UQ_bootstrap import *
from Full_UQ_bootstrap import *

from Plot_comparison_features import *


def Main_loop(Noise_Levels,DataSet_Name,Load_External_Data,Plot_Comparison_Figures,Full_Save,Models_to_run,Features,Bootstrap_Repetitions,PYCOL,Probabalistic_Classifiers,Bayesian_Classifiers,Scikit_Classifiers,Tenflow_Classifiers,Probflow_Classifiers):

    #where to save the data
    if Features==['Pri1','Pri2','Pri3']:
        Save_Name = 'Principal'
    else:
        Save_Name = 'Deviatoric'
    #(string) Folder to save data in

    #Run the thing
    for ii, Model in enumerate(Models_to_run):
        for jj, Noise_Level in enumerate(Noise_Levels):
            #Change the noise level to training and testing noise
            print('Noise level = ',Noise_Level)
            Training_noise = Noise_Level
            Testing_noise = Training_noise


            Savename = Save_Name+'_Tr'+str(Training_noise)+'_Te'+str(Testing_noise)



            #Read in the data
            Data = np.load('DataSets/'+DataSet_Name+'/DataSet.npy')
            Frequencies = np.genfromtxt('DataSets/'+DataSet_Name+'/Frequencies.csv',delimiter=',')
            Labels = np.genfromtxt('DataSets/'+DataSet_Name+'/Labels.csv',delimiter=',')
            Names = np.genfromtxt('DataSets/'+DataSet_Name+'/names.csv',delimiter=',',dtype=str)
            Number_Of_Classes = int(max(Labels)+1)

            #Object_list = []
            #with open('DataSets/'+DataSet_Name+'/Data_Overview.csv', newline='') as csvfile:
            #    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            #    for i,row in enumerate(spamreader):
            #        if i != 0:
            #            Object_list.append(row[0].split(',')[0])


            DataSet = np.load('DataSets/'+DataSet_Name+'/Descriptions.npy')
            Tensors = np.load('DataSets/'+DataSet_Name+'/Tensors.npy')


            Object_names = []
            Coin_Labels = []
            for name in Names:
                if name not in Coin_Labels:
                    Coin_Labels.append(name)
                    Object_names.append(name)

            #Number of Objects
            #Objects = int(max(Labels)+1)

            #Create the desired features
            Feature_Size = 0
            for Feat in Features:
                if Feat == 'Com':
                    Feature_Size += 1
                else:
                    Feature_Size += 2
            Feature_Data = np.zeros([np.shape(Data)[0],Feature_Size*len(Frequencies)])
            #Create a dictionary for Feature selection
            Feature_Dic = {'Eig1' : 0, 'Eig2' : 1, 'Eig3' : 2, 'Pri1' : 3, 'Pri2' : 4, 'Pri3' : 5, 'Dev2' : 6, 'Dev3' : 7, 'Com' : 8}

            #Create the Features and Labels to be used
            for i,Feature in enumerate(Features):
                Feature_Data[:,len(Frequencies)*2*i:len(Frequencies)*2*(i+1)] = Data[:,len(Frequencies)*2*Feature_Dic[Feature]:len(Frequencies)*2*(Feature_Dic[Feature]+1)]
            Labels = np.array([float(Labels[i]) for i in range(len(Labels))])

            #Create a way to keep track of the data
            Input_Array = np.zeros([len(Labels),Feature_Size*len(Frequencies)+2])
            Input_Array[:,0] = np.arange(len(Labels))
            Input_Array[:,1:-1] = Feature_Data
            Input_Array[:,-1] = Labels

            #Run bootstrap to get accurate result
            Results = np.zeros([Bootstrap_Repetitions])

            #Create a way to store the confusion matrices for later
            Con_mat_store = np.zeros([Number_Of_Classes,Number_Of_Classes,Bootstrap_Repetitions])

            #Create a way to keep track of the predictions and the correct predictions
            Actual = []
            Predictions = []
            Probabilities = []
            ProbabilitiesUp = []
            ProbabilitiesLow = []
            UQ = []
            PredictionsPL = []
            ActualPL = []
            ProbabilitiesPL = []
            ProbabilitiesUpPL = []
            ProbabilitiesLowPL = []
            UQPL = []

            #Make the folder
            try:
                if type(Testing_noise) == bool:
                    os.makedirs('Results/'+DataSet_Name+'/Noiseless/'+Model+'/'+Savename)
                else:
                    os.makedirs('Results/'+DataSet_Name+'/Noise_'+str(Testing_noise)+'/'+Model+'/'+Savename)
            except:
                pass

            # Here, we save Input_Array to disk.
            if Full_Save is True:
                try:
                    if type(Testing_noise) == bool:
                        np.savetxt(
                            'Results/' + DataSet_Name + '/Noiseless/' + Model + '/' + Savename + '/Input_Array.csv',
                            Input_Array, delimiter=',')
                    else:
                        np.savetxt('Results/' + DataSet_Name + '/Noise_' + str(
                            Testing_noise) + '/' + Model + '/' + Savename + '/Input_Array.csv',
                            Input_Array, delimiter=',')
                except:
                    pass

            Object_name_list = Object_names
            names = []
            numbers = []
            for i,name in enumerate(Names):
                if name.replace('_Orig','') not in names:
                    names.append(name.replace('_Orig',''))
                    numbers.append(Labels[i])

            reordered_names = [None]*len(names)

            for i, lab in enumerate(numbers):
                reordered_names[int(lab)] = names[i].replace('_',' ')

            #Split the Training and test data
            test_size = 0.25
            for k in range(Bootstrap_Repetitions):

                print(' Training model '+str((ii*len(Noise_Levels)*Bootstrap_Repetitions) + jj*Bootstrap_Repetitions + k + 1)+'/'+str(len(Models_to_run)*len(Noise_Levels)*Bootstrap_Repetitions),end='\r')
                #Split the training and test data
                np.random.shuffle(Input_Array)

                X_train, X_test, Y_train, Y_test = model_selection.train_test_split(Input_Array[:,:-1], Input_Array[:,-1], test_size=test_size, shuffle=False)


                #Add the noise to the training data
                if type(Training_noise)==bool:
                    X_train = X_train[:,1:]
                else:
                    X_train = Add_Noise(X_train,Training_noise,Tensors,Features,Frequencies,Feature_Dic)


                #Add the noise to the testing data
                if type(Testing_noise)==bool:
                    X_test = X_test[:,1:]
                else:
                    X_test = Add_Noise(X_test,Testing_noise,Tensors,Features,Frequencies,Feature_Dic)


                #Train the model

                #Do the preprocessing
                X_Means = np.mean(X_train,axis=0)
                X_SD = np.std(X_train,axis=0)

                # Saving Normalisation Coefficients:
                if Full_Save is True:
                    Normalisation_Coefficients = (X_Means, X_SD)
                    try:
                        if type(Testing_noise) == bool:
                            np.savetxt('Results/' + DataSet_Name + '/Noiseless/' + Model + '/' + Savename + '/Normalisation_Coefficients' + str(k)+'.csv',
                                       Normalisation_Coefficients, delimiter=',')
                        else:
                            np.savetxt('Results/' + DataSet_Name + '/Noise_' + str(Testing_noise) + '/' + Model + '/' + Savename + '/Normalisation_Coefficients' + str(k)+'.csv',
                                       Normalisation_Coefficients, delimiter=',')
                    except:
                        pass

                # Loading external data from disk. Requires that external_file_loader.py be run.
                if Load_External_Data == True:
                    try:
                        if type(Testing_noise) == bool:
                            input = np.genfromtxt('DataSets/'+DataSet_Name + '/X_Input.csv')
                        else:
                            input = np.genfromtxt('DataSets/'+DataSet_Name + '/X_Input.csv')
                    except:
                        pass

                    # if only one test class is used, we pad the array to a 1xN array.
                    if input.ndim == 1:
                        Y_test = np.zeros((1,)) + np.asarray(input[-1])
                        X_test = input[1:-1][None,:]
                    else:
                        Y_test = np.zeros((1,)) + np.asarray(input[:,-1])
                        X_test = input[:, 1:-1]

                # X_train_norm = X_train
                # X_test_norm = X_test
                X_train_norm = (X_train-X_Means)/X_SD
                X_test_norm = (X_test-X_Means)/X_SD

                # Plotting comparison figures. Currently this is only supported for one test class.
                if (k == 1) & (Plot_Comparison_Figures is True) & (Load_External_Data is True) & (X_test_norm.ndim == 1):
                    Plot_comparison_features(Frequencies,X_train_norm,Y_train)

                # Fit model on training set
                if Model in Scikit_Classifiers:
                    model = Train_Scikit(Model,X_train_norm,Y_train)
                elif Model in Tenflow_Classifiers:
                    model, probability_model = Train_Tenflow(Model,X_train_norm,Y_train)
                elif Model in Probflow_Classifiers:
                    model = Train_Probflow(Model,X_train_norm,Y_train,Number_Of_Classes)
                else:
                    print("Classifier not found")
                    
                    
                # save the model to disk
                if Full_Save is True:
                    try:
                        if type(Testing_noise) == bool:
                            joblib.dump(model, 'Results/' + DataSet_Name + '/Noiseless/' + Model + '/' + Savename + '/model' + str(k)+'.sav')
                        else:
                            joblib.dump(model, 'Results/' + DataSet_Name + '/Noise_' + str(Testing_noise) + '/' + Model + '/' + Savename + '/model' + str(k)+'.sav')
                    except:
                        pass

                #Obtain results
                if Model in Scikit_Classifiers:
                    Results,Con_mat_store,Predictions,Actual,PredictionsPL,ActualPL,Probabilities,ProbabilitiesPL,probs = Sklearn_results(k,model,X_test_norm,Y_test,Load_External_Data,Predictions,Actual,PredictionsPL,ActualPL,Probabalistic_Classifiers,Probabilities,ProbabilitiesPL,Results,Con_mat_store,Model)
                elif Model in Tenflow_Classifiers:
                    Results,Con_mat_store,Predictions,Actual,PredictionsPL,ActualPL,Probabilities,ProbabilitiesPL,probs = Tenflow_results(k,model,probability_model,X_test_norm,Y_test,Load_External_Data,Predictions,Actual,PredictionsPL,ActualPL,Probabalistic_Classifiers,Probabilities,ProbabilitiesPL,Results,Con_mat_store,Model)
                elif Model in Probflow_Classifiers:
                    Results,Con_mat_store,Predictions,Actual,PredictionsPL,ActualPL,Probabilities,ProbabilitiesPL,probs = Probflow_results(k,model,X_test_norm,Y_test,Load_External_Data,Predictions,Actual,PredictionsPL,ActualPL,Probabalistic_Classifiers,Probabilities,ProbabilitiesPL,Results,Con_mat_store,Model,Number_Of_Classes)    
    
                if (Model in Probabalistic_Classifiers): #or ('MLP' in Model):
                    ProbabilitiesUpPL,ProbabilitiesLowPL,UQPL = Simple_UQ_bootstrap(k,probs,Number_Of_Classes,ProbabilitiesUpPL,ProbabilitiesLowPL,UQPL)
                if (Model in Bayesian_Classifiers): #or ('MLP' in Model):
                    # included output of snapshots for testing
                   ProbabilitiesUpPL,ProbabilitiesLowPL,UQPL,ProbabilitiesPL = Full_UQ_bootstrap(model,k,probs,Number_Of_Classes,ProbabilitiesUpPL,ProbabilitiesLowPL,UQPL,X_test_norm,ProbabilitiesPL,DataSet_Name,Model,Savename,Testing_noise,Y_test,PYCOL,reordered_names)


 #          Saving score, prediction, and true class to disk.
            External_data_save(Load_External_Data,DataSet_Name,Model,Savename,Results,Predictions,Actual,Testing_noise)


            #Reorder
            # I think this creates a list of array, but order is not changed
            # Probabilities_appended looks strange. Recoded as stacked arrays (see above )
            Truth_list = [item for sublist in Actual for item in sublist]
            Prediction_list = [item for sublist in Predictions for item in sublist]
            Probabilities_appended = [item for sublist in Probabilities for item in sublist]

            # TODO add check for odd/even when calculating median.

            #Calculate the confidence of the predicitons
            if (Model in Probabalistic_Classifiers): #or ('MLP' in Model):
                Object_Confidence_Confidence,Object_Percentiles,Object_Confidence_Mean,Object_UQ_minval_val, Object_UQ_minval_low,Object_UQ_minval_up,Object_UQ_maxval_val,Object_UQ_maxval_low,Object_UQ_maxval_up,Bin_edges,Hist, Sorted_posteriors = Bootstrap_UQ(Number_Of_Classes,UQPL,ActualPL,ProbabilitiesPL,ProbabilitiesUpPL,ProbabilitiesLowPL)
##            elif (Model in Bayesian_Classifiers):
##                Object_Confidence_Confidence,Object_Percentiles,Object_Confidence_Mean,Object_UQ_minval_val, Object_UQ_minval_low,Object_UQ_minval_up,Object_UQ_maxval_val,Object_UQ_maxval_low,Object_UQ_maxval_up,Bin_edges,Hist, Sorted_posteriors = Bayesian_UQ(Number_Of_Classes,UQPL,ActualPL,ProbabilitiesPL,ProbabilitiesUpPL,ProbabilitiesLowPL)
            print("Completed predictions")

#           obtain confusion matrix
            My_confusion_matrix(Truth_list,Prediction_list,Testing_noise,DataSet_Name,Model,Savename)

#           create report and write report to disk
            Kappa = cohen_kappa_score(Truth_list,Prediction_list)
            Write_report(DataSet_Name,Model,Testing_noise,Savename,Kappa,Names,Truth_list,Prediction_list, Object_names,Labels)

#           plot results and save to disk for non probablistic classifiers.
            if (Model not in Bayesian_Classifiers):
                Plotting(Model,Probabalistic_Classifiers,Bayesian_Classifiers,Object_Confidence_Confidence,Object_Percentiles,DataSet_Name,Savename,Object_Confidence_Mean,Number_Of_Classes,PYCOL,Object_UQ_minval_val,Object_UQ_minval_low,Object_UQ_minval_up,Object_UQ_maxval_val,Object_UQ_maxval_low,Object_UQ_maxval_up,Bin_edges,Hist,Testing_noise,reordered_names)
                print("Completed plotting")

                if Bootstrap_Repetitions==1:
                   # For purpose of comparion provide a snapshot of the result obtained for the first of classification in each class.
                   # Do only for the first snapshot (if there are multiple)
                   # In this case ProbabilitiesUpPL, ProbabilitiesLowPL are prob estimate +/- UQPL - only contains value for single bootstap iteration
                    Output_snapshot(model,ProbabilitiesPL,ProbabilitiesUpPL,ProbabilitiesLowPL,UQPL,Y_test,Number_Of_Classes,PYCOL,reordered_names,DataSet_Name,Testing_noise,Model,Savename)
                    print("Completed snapshot classification")
