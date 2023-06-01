from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score
from scipy.stats import sem, t
import numpy as np

def Sklearn_results(k,model,X_test_norm,Y_test,Load_External_Data,Predictions,Actual,PredictionsPL,ActualPL,Probabalistic_Classifiers,Probabilities,ProbabilitiesPL,Results,Con_mat_store,Model):

    Results[k] = model.score(X_test_norm, Y_test)
    if Load_External_Data is False:
        Con_mat_store[:,:,k] = confusion_matrix(Y_test,model.predict(X_test_norm))

    Predictions.append(model.predict(X_test_norm))
    Actual.append(Y_test)
    if k==0:
        # Predictions are stacked vertically with the results of subsequent bootstrap iteraions under each other
        PredictionsPL = model.predict(X_test_norm)
        # True classes are appended horizontally with those of subsquent bootstrap iterations placed after each other
        ActualPL = Y_test
    else:
        PredictionsPL = np.vstack((PredictionsPL,model.predict(X_test_norm)))
        ActualPL = np.append(ActualPL, Y_test)

    if (Model in Probabalistic_Classifiers) or ('MLP' in Model):
        Probabilities.append(model.predict_proba(X_test_norm))
        probs = model.predict_proba(X_test_norm)
        if k == 0:
            ProbabilitiesPL = model.predict_proba(X_test_norm)
        else:
            ProbabilitiesPL = np.vstack((ProbabilitiesPL,model.predict_proba(X_test_norm)))
    return Results,Con_mat_store,Predictions,Actual,PredictionsPL,ActualPL,Probabilities,ProbabilitiesPL,probs
