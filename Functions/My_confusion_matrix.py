from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score
import numpy as np

def My_confusion_matrix(Truth_list,Prediction_list,Testing_noise,DataSet_Name,Model,Savename):
    Overall_Confusion_mat = confusion_matrix(Truth_list,Prediction_list)
    Overall_Confusion_mat = Overall_Confusion_mat / Overall_Confusion_mat.astype(float).sum(axis=1)
    
    if type(Testing_noise) == bool:
        np.savetxt('Results/'+DataSet_Name+'/Noiseless/'+Model+'/'+Savename+'/Confusion_Mat.csv',Overall_Confusion_mat,delimiter=',')
    else:
        np.savetxt('Results/'+DataSet_Name+'/Noise_'+str(Testing_noise)+'/'+Model+'/'+Savename+'/Confusion_Mat.csv',Overall_Confusion_mat,delimiter=',')


