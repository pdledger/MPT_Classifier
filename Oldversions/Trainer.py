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



#User Inputs

DataSet_Name = 'Class_8/Class_8_1000_Al_0.84_Sig_2.4'

# Option to load external testing data from disk. Requires that external_file_loader.py be run first.
Load_External_Data = False
# Option to plot comparison figures between the input array of simulated data and the external test data.
# Currently only supported for a single class test set.
Plot_Comparison_Figures = False
# Option to additionally save to disk: the model for each bootstrap iteration, the normalisation coefficients for each,
# bootstrap iteration, and the input array for each model, Used for debugging.
Full_Save = False

#Model to be used
#Optional models 'LogisticRegression', 'SVM', 'DecisionTree', 'RandomForest', 'GradientBoost', 'MLP','MLP,(n1,n2,...,nn)'

Models_to_run = ['LogisticRegression','SVM', 'DecisionTree', 'RandomForest', 'GradientBoost', 'MLP']
# Models_to_run = ['GradientBoost']
#Model = 'GradientBoost'
#(string) 'LogisticRegression', 'SVM', 'SVM-RBF', 'DecisionTree', 'RandomForest',
#'AdaBoost', 'GradientBoost', 'MLP' model to use when training the data



#Features
Features = ['Pri1','Pri2','Pri3']
#(list) list of features to be used options:
#'Eig1','Eig2','Eig3','Pri1','Pri2','Pri3','Dev2','Dev3','Com'
#Eigenvalues, Principal invarients, Deviatoric invarients, Comutator


#where to save the data
if Features==['Pri1','Pri2','Pri3']:
    Save_Name = 'Principal'
else:
    Save_Name = 'Deviatoric'
#(string) Folder to save data in



#How many times would you like to train the model
Bootstrap_Repetitions = 100
#(int) how many times to train the model to obtain an average accuracy


#Noise
Noise_Levels = [10,20,40]#,20,10]
#Noise_Levels = [False]
#(list) of (False or float or string)
#False if no noise to be added
#float in dB 20=> 10%, 40=> 1% actual noise
#string '75% x' for % of instances to add x noise to









########################################################################


#main script

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

sys.path.insert(0,"Functions")
from Noise_adder import *
from PreProcessors import *


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), sem(a)
    h = se * t.ppf((1 + confidence) / 2., n-1)
    return m, h


#Define the colours to use
PYCOL=['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf']

#Define the Probabalistic classifiers
Pobabalistic_Classifiers = ['LogisticRegression','RandomForest','AdaBoost','GradientBoost','MLP']








#Run the thing
for ii, Model in enumerate(Models_to_run):
    for jj, Noise_Level in enumerate(Noise_Levels):
        #Change the noise level to training and testing noise
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
                feature1_real = X_train_norm[:, 0:len(Frequencies)]
                feature1_imag = X_train_norm[:, len(Frequencies):2 * len(Frequencies)]

                feature2_real = X_train_norm[:, 2 * len(Frequencies):3 * len(Frequencies)]
                feature2_imag = X_train_norm[:, 3 * len(Frequencies):4 * len(Frequencies)]

                feature3_real = X_train_norm[:, 4 * len(Frequencies):5 * len(Frequencies)]
                feature3_imag = X_train_norm[:, 5 * len(Frequencies):6 * len(Frequencies)]

                object_class = Y_train[:, None] @ np.ones((1, len(Frequencies)))
                object_omega = np.ones((X_train_norm.shape[0], 1)) @ Frequencies[None, :]

                internal_data_dict = {'pri1_real': np.ravel(feature1_real), 'pri1_imag': np.ravel(feature1_imag),
                                      'pri2_real': np.ravel(feature2_real), 'pri2_imag': np.ravel(feature2_imag),
                                      'pri3_real': np.ravel(feature3_real), 'pri3_imag': np.ravel(feature3_imag),
                                      'omega': np.ravel(object_omega), 'class': np.ravel(object_class)}

                internal_dataframe = pd.DataFrame(internal_data_dict)

                feature1_real = X_test_norm[:, 0:len(Frequencies)]
                feature1_imag = X_test_norm[:, len(Frequencies):2 * len(Frequencies)]

                feature2_real = X_test_norm[:, 2 * len(Frequencies):3 * len(Frequencies)]
                feature2_imag = X_test_norm[:, 3 * len(Frequencies):4 * len(Frequencies)]

                feature3_real = X_test_norm[:, 4 * len(Frequencies):5 * len(Frequencies)]
                feature3_imag = X_test_norm[:, 5 * len(Frequencies):6 * len(Frequencies)]

                object_class = np.asarray(['External Data']* len(Frequencies))[:,None]
                object_omega = np.ones((X_test_norm.shape[0], 1)) @ Frequencies[None, :]

                external_data_dict = {'pri1_real': np.ravel(feature1_real), 'pri1_imag': np.ravel(feature1_imag),
                                      'pri2_real': np.ravel(feature2_real), 'pri2_imag': np.ravel(feature2_imag),
                                      'pri3_real': np.ravel(feature3_real), 'pri3_imag': np.ravel(feature3_imag),
                                      'omega': np.ravel(object_omega), 'class': np.ravel(object_class)}

                external_dataframe = pd.DataFrame(external_data_dict)
                total_dataframe = pd.concat([internal_dataframe, external_dataframe], ignore_index=True, sort=False)

                plt.figure()
                palette = sns.color_palette("Paired", 6)
                total_lineplot = sns.lineplot(data=total_dataframe, x='omega',
                                                 y='pri1_imag', hue='class', palette=palette)
                total_lineplot.set(xscale='log')

                plt.figure()
                total_lineplot = sns.lineplot(data=total_dataframe, x='omega',
                                                 y='pri1_real', hue='class', palette=palette)
                total_lineplot.set(xscale='log')

                plt.figure()
                total_lineplot = sns.lineplot(data=total_dataframe, x='omega',
                                              y='pri2_real', hue='class', palette=palette)
                total_lineplot.set(xscale='log')

                plt.figure()
                total_lineplot = sns.lineplot(data=total_dataframe, x='omega',
                                              y='pri2_imag', hue='class', palette=palette)
                total_lineplot.set(xscale='log')

                plt.figure()
                total_lineplot = sns.lineplot(data=total_dataframe, x='omega',
                                              y='pri3_real', hue='class', palette=palette)
                total_lineplot.set(xscale='log')

                plt.figure()
                total_lineplot = sns.lineplot(data=total_dataframe, x='omega',
                                              y='pri3_imag', hue='class', palette=palette)
                total_lineplot.set(xscale='log')

                plt.show()

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
            Results[k] = model.score(X_test_norm, Y_test)
            if Load_External_Data is False:
                Con_mat_store[:,:,k] = confusion_matrix(Y_test,model.predict(X_test_norm))
            Predictions.append(model.predict(X_test_norm))
            Actual.append(Y_test)

            if (Model in Pobabalistic_Classifiers) or ('MLP' in Model):
                Probabilities.append(model.predict_proba(X_test_norm))

        if Load_External_Data is True:
            # Saving score, prediction, and true class to disk.

            try:
                if type(Testing_noise) == bool:
                    np.savetxt('Results/' + DataSet_Name + '/Noiseless/' + Model + '/' + Savename + '/External_Results.csv',
                               Results, delimiter=',')
                    np.savetxt(
                        'Results/' + DataSet_Name + '/Noiseless/' + Model + '/' + Savename + '/External_Predictions.csv',
                        Predictions, delimiter=',')
                    np.savetxt(
                        'Results/' + DataSet_Name + '/Noiseless/' + Model + '/' + Savename + '/External_Actual.csv',
                        Actual, delimiter=',')
                else:
                    np.savetxt('Results/' + DataSet_Name + '/Noise_' + str(Testing_noise) + '/' + Model + '/' + Savename + '/External_Results.csv',
                               Results, delimiter=',')
                    np.savetxt(
                        'Results/' + DataSet_Name + '/Noise_' + str(Testing_noise) + '/' + Model + '/' + Savename + '/External_Predictions.csv',
                        Predictions, delimiter=',')
                    np.savetxt(
                        'Results/' + DataSet_Name + '/Noise_' + str(Testing_noise) + '/' + Model + '/' + Savename + '/External_Actual.csv',
                        Actual, delimiter=',')
            except:
                pass

        else:

            #Reorder
            Truth_list = [item for sublist in Actual for item in sublist]
            Prediction_list = [item for sublist in Predictions for item in sublist]
            Probabilities_appended = [item for sublist in Probabilities for item in sublist]


            #Calculate the confidence of the predicitons
            if (Model in Pobabalistic_Classifiers) or ('MLP' in Model):
                Object_Confidence_Mean = np.zeros([Number_Of_Classes,Number_Of_Classes])
                Object_Confidence_Confidence = np.zeros([Number_Of_Classes,Number_Of_Classes,4])
                Object_Percentiles = np.zeros([Number_Of_Classes,Number_Of_Classes,101])
                for i in range(Number_Of_Classes):
                    temp_probs = np.array([prob for j,prob in enumerate(Probabilities_appended) if Truth_list[j]==i])
                    Sorted_posteriors = np.zeros(np.shape(temp_probs))
                    for j in range(Number_Of_Classes):
                        Sorted_posteriors[:,j] = np.sort(temp_probs[:,j])
                        Object_Confidence_Confidence[i,j,0] = Sorted_posteriors[math.ceil(len(Sorted_posteriors)/40),j]
                        Object_Confidence_Confidence[i,j,1] = Sorted_posteriors[math.ceil(len(Sorted_posteriors)/4),j]
                        Object_Confidence_Mean[i,j] = Sorted_posteriors[math.ceil(len(Sorted_posteriors)/2),j]
                        Object_Confidence_Confidence[i,j,2] = Sorted_posteriors[math.floor(len(Sorted_posteriors)*3/4),j]
                        Object_Confidence_Confidence[i,j,3] = Sorted_posteriors[math.floor(len(Sorted_posteriors)*39/40),j]
                        for l in range(100):
                            Object_Percentiles[i,j,l] = Sorted_posteriors[math.ceil(len(Sorted_posteriors)*l/100),j]
                        Object_Percentiles[i,j,100] = Sorted_posteriors[-1,j]


            Overall_Confusion_mat = confusion_matrix(Truth_list,Prediction_list)
            Overall_Confusion_mat = Overall_Confusion_mat / Overall_Confusion_mat.astype(float).sum(axis=1)
            Kappa = cohen_kappa_score(Truth_list,Prediction_list)

            if type(Testing_noise) == bool:
                np.savetxt('Results/'+DataSet_Name+'/Noiseless/'+Model+'/'+Savename+'/Confusion_Mat.csv',Overall_Confusion_mat,delimiter=',')
            else:
                np.savetxt('Results/'+DataSet_Name+'/Noise_'+str(Testing_noise)+'/'+Model+'/'+Savename+'/Confusion_Mat.csv',Overall_Confusion_mat,delimiter=',')


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



            report = classification_report(Truth_list, Prediction_list, target_names=reordered_names)
            if  type(Testing_noise) == bool:
                f = open('Results/'+DataSet_Name+'/Noiseless/'+Model+'/'+Savename+'/Report.txt','w+')
                f1 = open('Results/'+DataSet_Name+'/Noiseless/'+Model+'/'+Savename+'/Kappa.txt','w+')
            else:
                f = open('Results/'+DataSet_Name+'/Noise_'+str(Testing_noise)+'/'+Model+'/'+Savename+'/Report.txt','w+')
                f1 = open('Results/'+DataSet_Name+'/Noise_'+str(Testing_noise)+'/'+Model+'/'+Savename+'/Kappa.txt','w+')
            f.write(report)
            f.close()
            f1.write(str(Kappa))
            f1.close()


            if (Model in Pobabalistic_Classifiers) or ('MLP' in Model):
                Object_Confidence_Confidence_saving = Object_Confidence_Confidence.reshape(Object_Confidence_Confidence.shape[0], -1)
                Object_Percentiles_saving = Object_Percentiles.reshape(Object_Percentiles.shape[0], -1)
                if type(Testing_noise) == bool:
                    np.savetxt('Results/'+DataSet_Name+'/Noiseless/'+Model+'/'+Savename+'/Posteriors_Mat.csv',Object_Confidence_Mean,delimiter=',')
                    np.savetxt('Results/'+DataSet_Name+'/Noiseless/'+Model+'/'+Savename+'/Confidence_Mat.csv',Object_Confidence_Confidence_saving,delimiter=',')
                    np.savetxt('Results/'+DataSet_Name+'/Noiseless/'+Model+'/'+Savename+'/Percentiles_Mat.csv',Object_Percentiles_saving,delimiter=',')
                else:
                    np.savetxt('Results/'+DataSet_Name+'/Noise_'+str(Testing_noise)+'/'+Model+'/'+Savename+'/Posteriors_Mat.csv',Object_Confidence_Mean,delimiter=',')
                    np.savetxt('Results/'+DataSet_Name+'/Noise_'+str(Testing_noise)+'/'+Model+'/'+Savename+'/Confidence_Mat.csv',Object_Confidence_Confidence_saving,delimiter=',')
                    np.savetxt('Results/'+DataSet_Name+'/Noise_'+str(Testing_noise)+'/'+Model+'/'+Savename+'/Percentiles_Mat.csv',Object_Percentiles_saving,delimiter=',')

                lims = np.ones([Number_Of_Classes],dtype=bool)
                labels = ['Probability','2.5%,97.5% Percentile']
                for i in range(Number_Of_Classes):
                    fig, ax = plt.subplots()
                    Bars = ax.bar(np.arange(Number_Of_Classes), Object_Confidence_Mean[i,:],color=[PYCOL[3] if np.argmax(Object_Confidence_Mean[i,:])==j else PYCOL[0] for j in range(Number_Of_Classes)], align='center', alpha=0.5, ecolor='black', capsize=10,label='Probability')

                    # including upper limits
                    for j in range(Number_Of_Classes):
                        ax.plot([j,j],[Object_Confidence_Confidence[i,j,0],Object_Confidence_Confidence[i,j,3]],linestyle='-', marker='_',markersize=16,color='k')

                    Legend_elements = [Patch(facecolor=PYCOL[0],alpha=0.6,label=r'Posterior probability $p(C_k|$data)'),Patch(facecolor=PYCOL[3],alpha=0.6,label=r'MAP estimate max$(p(C_k|$data))'),plt.Line2D([0],[0],color='k', marker='_', markersize=16 ,linestyle=' ', label='2.5,97.5 Percentile')]
                    ax.set_xticks(np.arange(Number_Of_Classes))
                    ax.set_xticklabels(reordered_names, rotation='vertical')
                    plt.subplots_adjust(bottom=0.25)
                    plt.gca().get_xticklabels()[i].set_fontweight('bold')
                    ax.yaxis.grid(True)
                    plt.ylim(0,1.05)
                    plt.xlabel(r'Classes $C_k$')
                    plt.ylabel(r'Posterior probability $p(C_k|$data)')
                    plt.legend(handles=Legend_elements)
                    if type(Testing_noise) == bool:
                        plt.savefig('Results/'+DataSet_Name+'/Noiseless/'+Model+'/'+Savename+'/figure'+str(i+1)+'.pdf')
                    else:
                        plt.savefig('Results/'+DataSet_Name+'/Noise_'+str(Testing_noise)+'/'+Model+'/'+Savename+'/figure'+str(i+1)+'.pdf')
                    plt.close()

                labels = ['Probability','Q1,Q3 Quartile']
                for i in range(Number_Of_Classes):
                    fig, ax = plt.subplots()
                    Bars = ax.bar(np.arange(Number_Of_Classes), Object_Confidence_Mean[i,:],color=[PYCOL[3] if np.argmax(Object_Confidence_Mean[i,:])==j else PYCOL[0] for j in range(Number_Of_Classes)], align='center', alpha=0.5, ecolor='black', capsize=10,label='Probability')

                    # including upper limits
                    for j in range(Number_Of_Classes):
                        ax.plot([j,j],[Object_Confidence_Confidence[i,j,1],Object_Confidence_Confidence[i,j,2]],linestyle='-', marker='_',markersize=16,color='k')

                    Legend_elements = [Patch(facecolor=PYCOL[0],alpha=0.6,label=r'Posterior probability $p(C_k|$data)'),Patch(facecolor=PYCOL[3],alpha=0.6,label=r'MAP estimate max$(p(C_k|$data))'),plt.Line2D([0],[0],color='k', marker='_', markersize=16 ,linestyle=' ', label='Q1,Q3 Quartile')]
                    ax.set_xticks(np.arange(Number_Of_Classes))
                    ax.set_xticklabels(reordered_names, rotation='vertical')
                    plt.subplots_adjust(bottom=0.25)
                    plt.gca().get_xticklabels()[i].set_fontweight('bold')
                    ax.yaxis.grid(True)
                    plt.ylim(0,1.05)
                    plt.xlabel(r'Classes $C_k$')
                    plt.ylabel(r'Posterior probability $p(C_k|$data)')
                    plt.legend(handles=Legend_elements)
                    if type(Testing_noise) == bool:
                        plt.savefig('Results/'+DataSet_Name+'/Noiseless/'+Model+'/'+Savename+'/figure'+str(i+Number_Of_Classes+1)+'.pdf')
                    else:
                        plt.savefig('Results/'+DataSet_Name+'/Noise_'+str(Testing_noise)+'/'+Model+'/'+Savename+'/figure'+str(i+Number_Of_Classes+1)+'.pdf')
                    plt.close()
