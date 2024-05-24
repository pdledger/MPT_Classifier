import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
import seaborn as sns
import time

def Plot_principal_features(Frequencies,X_train_norm,X_test_norm,Y_train,DataSet_Name,Model,Savename,Testing_noise,reordered_names):

    # We have applied SVD to reduce the number of features
    feature1 = X_train_norm[:,0] # first feature for the full dataset
    feature2 = X_train_norm[:,1] # second feature for the full dataset
    print(np.shape(feature1))

##        
##    feature1_real = X_train_norm[:, 0:len(Frequencies)]
##    feature1_imag = X_train_norm[:, len(Frequencies):2 * len(Frequencies)]
##
####    plt.figure()
####    dum,f = np.shape(feature1_imag)
####    print(dum,f,len(Frequencies))
####    for i in range(dum):
####        plt.semilogx((Frequencies),feature1_real[i,:])
####
####    plt.show()
##
##    feature2_real = X_train_norm[:, 2 * len(Frequencies):3 * len(Frequencies)]
##    feature2_imag = X_train_norm[:, 3 * len(Frequencies):4 * len(Frequencies)]
##
##    feature3_real = X_train_norm[:, 4 * len(Frequencies):5 * len(Frequencies)]
##    feature3_imag = X_train_norm[:, 5 * len(Frequencies):6 * len(Frequencies)]

    object_class = Y_train[:, None] #@ np.ones((1, len(Frequencies)))

    #object_omega = np.ones((X_train_norm.shape[0], 1)) @ Frequencies[None, :]
    # Create a list of labels for the plots
    labels=[]
    for cl in np.ravel(object_class):
        labels.append(reordered_names[int(cl)])


    internal_data_dict = {'Reduced Feature 1': np.ravel(feature1), 'Reduced Feature 2': np.ravel(feature2),
                         'class': np.ravel(object_class),'Class Names':labels}

    #print(np.shape(np.ravel(feature1_real)), np.shape(np.ravel(feature1_imag)), np.shape(np.ravel(feature2_real)),np.shape(np.ravel(feature2_imag)),np.shape(np.ravel(feature3_real)),np.shape(np.ravel(feature3_imag)),np.shape(np.ravel(object_omega)),np.shape( np.ravel(object_class)))
    internal_dataframe = pd.DataFrame(internal_data_dict)

    plt.figure()
    palette = sns.color_palette()#"Paired", 6)
    sactter_plot = sns.scatterplot(data=internal_dataframe, x='Reduced Feature 1', y = 'Reduced Feature 2', hue='Class Names', palette=palette, hue_order=reordered_names)


    if type(Testing_noise) == bool:
        plt.savefig('Results/'+DataSet_Name+'/Noiseless/'+Model+'/'+Savename+'/Feat1_Feat2_angles.pdf')
    else:
        plt.savefig('Results/'+DataSet_Name+'/Noise_'+str(Testing_noise)+'/'+Model+'/'+Savename+'/Feat1_Feat2_angles.pdf')
    plt.close()

    #plt.show()
