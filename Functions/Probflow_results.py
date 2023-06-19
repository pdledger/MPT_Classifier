from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score
from scipy.stats import sem, t
import numpy as np
import time

def Probflow_results(k,model,X_test_norm,Y_test,Load_External_Data,Predictions,Actual,PredictionsPL,ActualPL,Probabalistic_Classifiers,Probabilities,ProbabilitiesPL,Results,Con_mat_store,Model,Number_Of_Classes):
    test_acc=model.metric('accuracy',X_test_norm,Y_test)
    Results[k] = test_acc

    dum = np.shape(X_test_norm)
    N = dum[0] # Number of items in the data set
    probs=np.array([],dtype=float)
    for myk in range(Number_Of_Classes):
        kclass = myk*np.ones(N).astype('float32')
        probk = model.prob(X_test_norm.astype('float32'),kclass)
        # model.prob gives the MAP estimate for P(C=k| X_test_norm) for all
        # each instance of the data so we need to repeat for each class and
        # arrange data as columns in data
        if myk==0:
            probs = probk
        else:
            probs = np.vstack((probs,probk))
        
    # probs has dimension Number_Of_Classes x N we want
    # it to be stored as N x Number_Of_Classes
    probs = np.transpose(probs)
    
    Probabilities.append(probs)
    # obtain predictions
    dum = np.shape(X_test_norm)
    predictions=[]
    for n in range(dum[0]):
        case = probs[n]
        case_class = np.argmax(case)
        predictions.append(case_class)
    Actual.append(Y_test)
    #print(predictions,model.predict(X_test_norm,method="mean"))
    
    Predictions.append(predictions)
    if k==0:
        # Predictions are stacked vertically with the results of subsequent bootstrap iteraions under each other
        PredictionsPL =predictions
        # True classes are appended horizontally with those of subsquent bootstrap iterations placed after each other
        ActualPL = Y_test
    else:
        PredictionsPL = np.vstack((PredictionsPL,predictions))
        ActualPL = np.append(ActualPL, Y_test)
    if k == 0:
        ProbabilitiesPL = probs
    else:
        ProbabilitiesPL = np.vstack((ProbabilitiesPL,probs))
  
    if Load_External_Data is False:
        Con_mat_store[:,:,k] = confusion_matrix(Y_test,predictions)

    return Results,Con_mat_store,Predictions,Actual,PredictionsPL,ActualPL,Probabilities,ProbabilitiesPL,probs
