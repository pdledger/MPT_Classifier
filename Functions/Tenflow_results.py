from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score
from scipy.stats import sem, t
import numpy as np

def Tenflow_results(k,model,probability_model,X_test_norm,Y_test,Load_External_Data,Predictions,Actual,PredictionsPL,ActualPL,Probabalistic_Classifiers,Probabilities,ProbabilitiesPL,Results,Con_mat_store,Model):
    test_loss, test_acc = model.evaluate(X_test_norm,  Y_test, verbose=2)
    Results[k] = test_acc

    probs =  probability_model.predict(X_test_norm)
    Probabilities.append(probs)
    # obtain predictions
    dum = np.shape(X_test_norm)
    predictions=[]
    for n in range(dum[0]):
        case = probs[n]
        case_class = np.argmax(case)
        predictions.append(case_class)
    Actual.append(Y_test)
    print(predictions)
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
