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
from Jensen_Shannon import *

from Plot_comparison_features import *
from Plot_principal_features import *

from ApplySVD import *

from Check_KL_divergence import *

def Main_loop(Noise_Levels,DataSet_Name,Load_External_Data,Plot_Comparison_Figures,Full_Save,Models_to_run,Features,Bootstrap_Repetitions,
              PYCOL,Probabalistic_Classifiers,Bayesian_Classifiers,Scikit_Classifiers,Tenflow_Classifiers,Probflow_Classifiers,Reduce_Features,
              Trainer_Settings,Plot_Principal_Componenets):

    #where to save the data
    if Features==['Pri1','Pri2','Pri3']:
        Save_Name = 'Principal'
    elif Features == ['Eig1', 'Eig2', 'Eig3']:
        Save_Name = 'Eigenvalues'
    else:
        Save_Name = 'Deviatoric'
    #(string) Folder to save data in

    for jj, Noise_Level in enumerate(Noise_Levels):
        #Change the noise level to training and testing noise
        print('Noise level = ',Noise_Level)
        Training_noise = Noise_Level
        Testing_noise = Training_noise

        #Read in the data
        Data = np.load('DataSets/'+DataSet_Name+'/DataSet.npy')
        print(np.shape(Data))
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

        # Check if the list of Features contains angles
        # Only read the angle data if we are going to use this feature.
        AngleFlag = "Off"
        for Feat in Features:
            if Feat == 'AngleRtildeI':
                AngleFlag = "On"
        if AngleFlag== "On":
            Angles = np.load('DataSets/'+DataSet_Name+'/Angles.npy')
        else:
            Angles=[]



        Object_names = []
        Coin_Labels = []
        for name in Names:
            #print(name)
            if name not in Coin_Labels:
                Coin_Labels.append(name)
                Object_names.append(name)

        #Number of Objects
        #Objects = int(max(Labels)+1)

        #Create the desired features
        Feature_Size = 0
        for Feat in Features:
            if Feat == 'Com' or Feat == 'AngleRtildeI':
                Feature_Size += 1 # These features are real valued only
            else:
                Feature_Size += 2 # Other features are complex and so we need to consider real and imaginary parts
        Feature_Data = np.zeros([np.shape(Data)[0],Feature_Size*len(Frequencies)])
        #Create a dictionary for Feature selection
        Feature_Dic = {'Eig1' : 0, 'Eig2' : 1, 'Eig3' : 2, 'Pri1' : 3, 'Pri2' : 4, 'Pri3' : 5, 'Dev2' : 6, 'Dev3' : 7, 'Com' : 8, 'AngleRtildeI': 9}

        #Create the Features and Labels to be used
        count=0
        # Code updated since we are not always dealing with real and imaginary parts for
        # Feature_Dic[Feature]=8,9 then we have only a real part.
        for i,Feature in enumerate(Features):
            #print(Feature,Feature_Dic[Feature],len(Frequencies)*2*Feature_Dic[Feature],len(Frequencies)*2*(Feature_Dic[Feature]+1))
            if Feature_Dic[Feature] < 8:
        #    Feature_Data[:,len(Frequencies)*2*i:len(Frequencies)*2*(i+1)] = Data[:,len(Frequencies)*2*Feature_Dic[Feature]:len(Frequencies)*2*(Feature_Dic[Feature]+1)]
                Feature_Data[:,count:count+len(Frequencies)*2] = Data[:,len(Frequencies)*2*Feature_Dic[Feature]:len(Frequencies)*2*(Feature_Dic[Feature]+1)]
                count+=len(Frequencies)*2
            elif Feature_Dic[Feature]==8:
                Feature_Data[:,count:count+len(Frequencies)]=Data[:,len(Frequencies)*2*Feature_Dic[Feature]:len(Frequencies)*2*(Feature_Dic[Feature])+len(Frequencies)]
                count+=len(Frequencies)
            elif Feature_Dic[Feature]==9:
                Feature_Data[:,count:count+len(Frequencies)]=Data[:,len(Frequencies)*2*(Feature_Dic[Feature]-1)+len(Frequencies):len(Frequencies)*2*(Feature_Dic[Feature]-1)+2*len(Frequencies)]
                count+=len(Frequencies)


        Labels = np.array([float(Labels[i]) for i in range(len(Labels))])

        #print(np.shape(Feature_Data))
        #print(Feature_Data[:,len(Frequencies)])


        #Create a way to keep track of the data
        Input_Array = np.zeros([len(Labels),Feature_Size*len(Frequencies)+2])
        print(np.shape(Input_Array),np.shape(Feature_Data))
        print("Data",Data)
        Input_Array[:,0] = np.arange(len(Labels))
        Input_Array[:,1:-1] = Feature_Data
        Input_Array[:,-1] = Labels
        # The first column of Input_Array is  0,1,len(labels).
        # The next Feature_Size*len(Frequencies) contain the data.
        # The last column are the Labels.


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
        for k in range(1):#range(Bootstrap_Repetitions):
            # Loop over bootstrap iterations and then methods does not make sense.
            # But we also want to check the methods for the same data. Disabled for now.
            #print(' Training model '+str((ii*len(Noise_Levels)*Bootstrap_Repetitions) + jj*Bootstrap_Repetitions + k + 1)+'/'+str(len(Models_to_run)*len(Noise_Levels)*Bootstrap_Repetitions),end='\r')
            #Split the training and test data
            np.random.shuffle(Input_Array)

            X_train, X_test, Y_train, Y_test = model_selection.train_test_split(Input_Array[:,:-1], Input_Array[:,-1], test_size=test_size, shuffle=False)
            # Recall that the first column of Input_Array is  0,1,len(labels).
            # The next Feature_Size*len(Frequencies) columns are the data.
            # The last column are the Labels.

            # X_train and X_test come from sampling rows of Input_Array[:,:-1] and so their first column will still be 0,1,len(labels).
            # the next columns are the data we want.
            # Y_train and Y_test come from sampling (same) rows of Input_Array[:,-1] and so contain the labels.



            #Add the noise to the training data
            # Note that this removes the first column of X_train
            if type(Training_noise)==bool:
                X_train = X_train[:,1:]
            else:
                print('to here')
                X_train = Add_Noise(X_train,Training_noise,Tensors,Features,Frequencies,Feature_Dic,Angles,AngleFlag)



            #Add the noise to the testing data
            # Note that this removes the first column of X_test
            if type(Testing_noise)==bool:
                X_test = X_test[:,1:]
            else:
                X_test = Add_Noise(X_test,Testing_noise,Tensors,Features,Frequencies,Feature_Dic,Angles,AngleFlag)

            for ii, Model in enumerate(Models_to_run):
                print(Model)
                if Model in Probflow_Classifiers:
                    # Set HyperParameters for Inverse Gamma Distributions
                    # Choosing Alpha, Beta small should result in an non-informative prior
                    # However in practice there is a limit as to how small they can be chosen.
                    # Some recommend ~ 1, others 0.1, 0.01 or 0.001
                    AlphaList = np.array([1, 0.1, 0.01])
                    BetaList = np.array([1, 0.1, 0.01])
                else:
                    AlphaList = np.array([1])
                    BetaList = np.array([1])


 #              In the case of Probablistic classifiers we use the InverseGamma distribution
 #              Choose possible Alpha, Beta so we have a non-informative Prior
                Vary_alpha_beta=[]
                for kk in range(len(AlphaList)):
                    for ll in range(1):#len(BetaList)):
                        #dummy second loop as we only use kk
                        Alpha = AlphaList[kk]
                        Beta = BetaList[kk]


                        Savename = Save_Name+'_Tr'+str(Training_noise)+'_Te'+str(Testing_noise)+'alpha_'+str(Alpha)+'beta_'+str(Beta)
                        #Make the folder
                        try:
                            if type(Testing_noise) == bool:
                                os.makedirs('Results/'+DataSet_Name+'/Noiseless/'+Model+'/'+Savename)
                            else:
                                os.makedirs('Results/'+DataSet_Name+'/Noise_'+str(Testing_noise)+'/'+Model+'/'+Savename)
                        except:
                            pass
                        if type(Testing_noise) == bool:
                            with open('Results/'+DataSet_Name+'/Noiseless/'+Model+'/'+Savename+'/Trainer_Settings.csv', 'w') as fset:
                                fset.write(str(np.asarray(Trainer_Settings)))
                        else:
                            #Print a copy of the Settings used as a record
                            with open('Results/'+DataSet_Name+'/Noise_'+str(Testing_noise)+'/'+Model+'/'+Savename+'/Trainer_Settings.csv', 'w') as fset:
                                fset.write(str(np.asarray(Trainer_Settings)))

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

                        #X_train_norm = (X_train-X_Means)/X_SD
                        #X_test_norm = (X_test-X_Means)/X_SD


                        # Give an option to create a reduced number of features by applying SVD
                        if Reduce_Features == True:
                            # Subtract means before SVD
                            X_train = X_train- X_Means
                            X_test = X_test - X_Means

                            X_train_norm, X_test_norm = ApplySVD (X_test,X_train, Testing_noise,DataSet_Name,Model,Savename )

                        else:
                            X_train_norm = (X_train-X_Means)/X_SD
                            X_test_norm = (X_test-X_Means)/X_SD



                        # Use KL-divergence
                        #Check_KL_divergence(X_train_norm,Y_train)


                        # Plotting comparison figures. Currently this is only supported for one test class.
                        if (k==0) & (Plot_Principal_Componenets is True) & (Reduce_Features is True) :
                            # Make a plot of first two principal components
                            Plot_principal_features(Frequencies,X_train_norm,X_test_norm,Y_train,DataSet_Name,Model,Savename,Testing_noise,reordered_names)

                        #if (k == 1) & (Plot_Comparison_Figures is True) & (Load_External_Data is True) :#& (X_test_norm.ndim == 1):
                        if (k == 0) & (Plot_Comparison_Figures is True) & (Reduce_Features is False):
                            # This will only work without SVD being applied
                            Plot_comparison_features(Frequencies,X_train,X_test,Y_train,DataSet_Name,Model,Savename,Testing_noise,Features,Load_External_Data,reordered_names)

                        if Plot_Comparison_Figures is False:

                            # Fit model on training set
                            if Model in Scikit_Classifiers:
                                model = Train_Scikit(Model,X_train_norm,Y_train)
                            elif Model in Tenflow_Classifiers:
                                model, probability_model = Train_Tenflow(Model,X_train_norm,Y_train)
                            elif Model in Probflow_Classifiers:
                                model = Train_Probflow(Model,X_train_norm,Y_train,Number_Of_Classes,Alpha,Beta)
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
                               ProbabilitiesUpPL,ProbabilitiesLowPL,UQPL,ProbabilitiesPL,df = Full_UQ_bootstrap(model,k,probs,Number_Of_Classes,ProbabilitiesUpPL,ProbabilitiesLowPL,UQPL,X_test_norm,ProbabilitiesPL,DataSet_Name,Model,Savename,Testing_noise,Y_test,PYCOL,reordered_names,Alpha,Beta,Load_External_Data)
                               Vary_alpha_beta.append(df)

                    if Plot_Comparison_Figures is False:
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


                # Create plots of Kolmogorov-Smirnov tests to compare results for different alpha,beta
                if (Model in Probflow_Classifiers) and (Plot_Comparison_Figures is False):
                    Jensen_Shannon(Vary_alpha_beta,AlphaList,BetaList,Number_Of_Classes,reordered_names,DataSet_Name,Testing_noise,Model,Savename)
