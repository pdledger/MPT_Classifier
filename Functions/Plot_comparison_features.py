import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
import seaborn as sns
import time

def Plot_comparison_features(Frequencies,X_train_norm,X_test_norm,Y_train,DataSet_Name,Model,Savename,Testing_noise,\
                             Features,Load_External_Data,reordered_names):

    if Features==['Pri1','Pri2','Pri3']:    
        feature1_real = X_train_norm[:, 0:len(Frequencies)]
        feature1_imag = X_train_norm[:, len(Frequencies):2 * len(Frequencies)]

    ##    plt.figure()
    ##    dum,f = np.shape(feature1_imag)
    ##    print(dum,f,len(Frequencies))
    ##    for i in range(dum):
    ##        plt.semilogx((Frequencies),feature1_real[i,:])
    ##
    ##    plt.show()

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

        #print(np.shape(np.ravel(feature1_real)), np.shape(np.ravel(feature1_imag)), np.shape(np.ravel(feature2_real)),np.shape(np.ravel(feature2_imag)),np.shape(np.ravel(feature3_real)),np.shape(np.ravel(feature3_imag)),np.shape(np.ravel(object_omega)),np.shape( np.ravel(object_class)))
        internal_dataframe = pd.DataFrame(internal_data_dict)

    ##    plt.figure()
    ##    palette = sns.color_palette("Paired", 6)
    ##    total_lineplot = sns.lineplot(data=internal_dataframe, x='omega',
    ##                                     y='pri1_imag', hue='class', palette=palette)
    ##    total_lineplot.set(xscale='log')
    ##    plt.figure()
    ##    total_lineplot = sns.lineplot(data=internal_dataframe, x='omega',
    ##                                     y='pri1_real', hue='class', palette=palette)
    ##    total_lineplot.set(xscale='log')
    ##
    ##    plt.figure()
    ##    total_lineplot = sns.lineplot(data=internal_dataframe, x='omega',
    ##                                     y='pri2_imag', hue='class', palette=palette)
    ##    total_lineplot.set(xscale='log')
    ##    plt.figure()
    ##    total_lineplot = sns.lineplot(data=internal_dataframe, x='omega',
    ##                                     y='pri2_real', hue='class', palette=palette)
    ##    total_lineplot.set(xscale='log')
    ##
    ##    plt.figure()
    ##    total_lineplot = sns.lineplot(data=internal_dataframe, x='omega',
    ##                                     y='pri3_imag', hue='class', palette=palette)
    ##    total_lineplot.set(xscale='log')
    ##    plt.figure()
    ##    total_lineplot = sns.lineplot(data=internal_dataframe, x='omega',
    ##                                     y='pri3_real', hue='class', palette=palette)
    ##    total_lineplot.set(xscale='log')
    ##    time.sleep(10)

        if Load_External_Data==True:
            feature1_real = X_test_norm[:, 0:len(Frequencies)]
            feature1_imag = X_test_norm[:, len(Frequencies):2 * len(Frequencies)]

            
            feature2_real = X_test_norm[:, 2 * len(Frequencies):3 * len(Frequencies)]
            feature2_imag = X_test_norm[:, 3 * len(Frequencies):4 * len(Frequencies)]

            feature3_real = X_test_norm[:, 4 * len(Frequencies):5 * len(Frequencies)]
            feature3_imag = X_test_norm[:, 5 * len(Frequencies):6 * len(Frequencies)]

            object_class = np.asarray(['External Data']* len(Frequencies))[:,None]
            object_omega = np.ones((X_test_norm.shape[0], 1)) @ Frequencies[None, :]

            print(np.shape(np.ravel(feature1_real)),np.shape(np.ravel(feature1_imag)),np.shape(np.ravel(feature2_real)),
                  np.shape(np.ravel(feature2_imag)),np.shape(np.ravel(feature3_real)),np.shape(np.ravel(feature3_imag)),
                  np.shape(np.ravel(object_omega)),np.shape(np.ravel(object_class)))

            external_data_dict = {'pri1_real': np.ravel(feature1_real), 'pri1_imag': np.ravel(feature1_imag),
                                  'pri2_real': np.ravel(feature2_real), 'pri2_imag': np.ravel(feature2_imag),
                                  'pri3_real': np.ravel(feature3_real), 'pri3_imag': np.ravel(feature3_imag),
                                  'omega': np.ravel(object_omega), 'class': np.ravel(object_class)}

            external_dataframe = pd.DataFrame(external_data_dict)
            total_dataframe = pd.concat([internal_dataframe, external_dataframe], ignore_index=True, sort=False)
        else:
            total_dataframe = internal_dataframe

        plt.figure()
        palette = sns.color_palette("Paired", 6)
        total_lineplot = sns.lineplot(data=total_dataframe, x='omega',
                                         y='pri1_imag', hue='class', palette=palette)
        total_lineplot.set(xscale='log')
        if type(Testing_noise) == bool:
            plt.savefig('Results/'+DataSet_Name+'/Noiseless/'+Model+'/'+Savename+'/datasetpri1_imag.pdf')
        else:
            plt.savefig('Results/'+DataSet_Name+'/Noise_'+str(Testing_noise)+'/'+Model+'/'+Savename+'/datasetpri1_imag.pdf')
        plt.close()

        plt.figure()
        total_lineplot = sns.lineplot(data=total_dataframe, x='omega',
                                         y='pri1_real', hue='class', palette=palette)
        total_lineplot.set(xscale='log')
        if type(Testing_noise) == bool:
            plt.savefig('Results/'+DataSet_Name+'/Noiseless/'+Model+'/'+Savename+'/datasetpri1_real.pdf')
        else:
            plt.savefig('Results/'+DataSet_Name+'/Noise_'+str(Testing_noise)+'/'+Model+'/'+Savename+'/datasetpri1_real.pdf')
        plt.close()

        plt.figure()
        total_lineplot = sns.lineplot(data=total_dataframe, x='omega',
                                      y='pri2_real', hue='class', palette=palette)
        total_lineplot.set(xscale='log')
        if type(Testing_noise) == bool:
            plt.savefig('Results/'+DataSet_Name+'/Noiseless/'+Model+'/'+Savename+'/datasetpri2_real.pdf')
        else:
            plt.savefig('Results/'+DataSet_Name+'/Noise_'+str(Testing_noise)+'/'+Model+'/'+Savename+'/datasetpri2_real.pdf')
        plt.close()



        plt.figure()
        total_lineplot = sns.lineplot(data=total_dataframe, x='omega',
                                      y='pri2_imag', hue='class', palette=palette)
        total_lineplot.set(xscale='log')
        if type(Testing_noise) == bool:
            plt.savefig('Results/'+DataSet_Name+'/Noiseless/'+Model+'/'+Savename+'/datasetpri2_imag.pdf')
        else:
            plt.savefig('Results/'+DataSet_Name+'/Noise_'+str(Testing_noise)+'/'+Model+'/'+Savename+'/datasetpri2_imag.pdf')
        plt.close()


        plt.figure()
        total_lineplot = sns.lineplot(data=total_dataframe, x='omega',
                                      y='pri3_real', hue='class', palette=palette)
        total_lineplot.set(xscale='log')
        if type(Testing_noise) == bool:
            plt.savefig('Results/'+DataSet_Name+'/Noiseless/'+Model+'/'+Savename+'/datasetpri3_real.pdf')
        else:
            plt.savefig('Results/'+DataSet_Name+'/Noise_'+str(Testing_noise)+'/'+Model+'/'+Savename+'/datasetpri3_real.pdf')
        plt.close()



        plt.figure()
        total_lineplot = sns.lineplot(data=total_dataframe, x='omega',
                                      y='pri3_imag', hue='class', palette=palette)
        total_lineplot.set(xscale='log')
        if type(Testing_noise) == bool:
            plt.savefig('Results/'+DataSet_Name+'/Noiseless/'+Model+'/'+Savename+'/datasetpri3_imag.pdf')
        else:
            plt.savefig('Results/'+DataSet_Name+'/Noise_'+str(Testing_noise)+'/'+Model+'/'+Savename+'/datasetpri3_imag.pdf')
        plt.close()

        #plt.show()
    elif Features==['AngleRtildeI']:
        feature1_angles = X_train_norm[:, 0:len(Frequencies)]
        object_class = Y_train[:, None] @ np.ones((1, len(Frequencies)))
        object_omega = np.ones((X_train_norm.shape[0], 1)) @ Frequencies[None, :]
        labels=[]
        for cl in np.ravel(object_class):
            labels.append(reordered_names[int(cl)])

        internal_data_dict = {'d_R(Rtilde,I)': np.ravel(feature1_angles),
                              'omega': np.ravel(object_omega), 'class': np.ravel(object_class),'Class Names':labels}
        internal_dataframe = pd.DataFrame(internal_data_dict)
        plt.figure()
        palette = sns.color_palette()
        total_lineplot = sns.lineplot(data=internal_dataframe, x='omega',
                                         y='d_R(Rtilde,I)', hue='Class Names', palette=palette,hue_order=reordered_names )
        total_lineplot.set(xscale='log')
        plt.ylim(0,0.12)
        if type(Testing_noise) == bool:
            plt.savefig('Results/'+DataSet_Name+'/Noiseless/'+Model+'/'+Savename+'/d_R(Rtilde,I).pdf')
        else:
            plt.savefig('Results/'+DataSet_Name+'/Noise_'+str(Testing_noise)+'/'+Model+'/'+Savename+'/d_R(Rtilde,I).pdf')
        plt.close()

        

        #plt.show()
