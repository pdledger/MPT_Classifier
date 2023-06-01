#Import
import os
import sys
from sys import exit
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score


def Write_report(DataSet_Name,Model,Testing_noise,Savename,Kappa,Names,Truth_list,Prediction_list, Object_names,Labels):

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
    return reordered_names

