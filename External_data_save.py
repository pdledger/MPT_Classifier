import numpy as np

def External_data_save(Load_External_Data,DataSet_Name,Model,Savename,Results,Predictions,Actual,Testing_noise):
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
