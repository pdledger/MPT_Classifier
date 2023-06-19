import os
import sys
from sys import exit
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

def Output_snapshot(model,ProbabilitiesPL,ProbabilitiesUpPL,ProbabilitiesLowPL,UQPL,Y_test,Number_Of_Classes,PYCOL,reordered_names,DataSet_Name,Testing_noise,Model,Savename):

    # determine indices of those outputs to show - by default first
        # test result for each class.

    N = np.shape(Y_test)[0]
    print("Found",N," Results")
    print(np.shape(ProbabilitiesPL),np.shape(ProbabilitiesUpPL),np.shape(ProbabilitiesLowPL))
    # This function is only called if there is a single bootstap iteration and so we only need loop at the first instance

    
    for k in range(Number_Of_Classes):
        flag=0
        if flag== 0:
            for n in range(N):
                if np.abs(Y_test[n]-k) < 1e-10:
                    # found first instance of class
                    flag=1
                    # Plot out bar graphs showing output for the uq value associated with this output
                    labels = ['Probability','UQ min']
                    fig, ax = plt.subplots()
                    Bars = ax.bar(np.arange(Number_Of_Classes), ProbabilitiesPL[n,:],color=[PYCOL[3] if np.argmax(ProbabilitiesPL[n,:])==j else PYCOL[0] for j in range(Number_Of_Classes)], align='center', alpha=0.5, ecolor='black', capsize=10,label='Probability')

                    # including upper limits
                    for j in range(Number_Of_Classes):
                        ax.plot([j,j],[ProbabilitiesLowPL[n,j],ProbabilitiesUpPL[n,j]],linestyle='-', marker='_',markersize=16,color='k')

                    Legend_elements = [Patch(facecolor=PYCOL[0],alpha=0.6,label=r'Mean posterior probability $p(C_k|$data)'),
                                       Patch(facecolor=PYCOL[3],alpha=0.6,label=r'Mean posterior probability max$(p(C_k|$data))'),
                                       plt.Line2D([0],[0],color='k', marker='_', markersize=16 ,linestyle=' ', label='95% Confidence Interval')]
                    ax.set_xticks(np.arange(Number_Of_Classes))
                    ax.set_xticklabels(reordered_names, rotation='vertical')
                    plt.subplots_adjust(bottom=0.25)
                    plt.gca().get_xticklabels()[k].set_fontweight('bold')
                    ax.yaxis.grid(True)
                    plt.ylim(0,1.05)
                    plt.xlabel(r'Classes $C_k$')
                    plt.ylabel(r'Posterior probability $p(C_k|$data)')
                    plt.legend(handles=Legend_elements)
                    if type(Testing_noise) == bool:
                        plt.savefig('Results/'+DataSet_Name+'/Noiseless/'+Model+'/'+Savename+'/snapshotfigure'+str(k)+'.pdf')
                    else:
                        plt.savefig('Results/'+DataSet_Name+'/Noise_'+str(Testing_noise)+'/'+Model+'/'+Savename+'/snapshotfigure'+str(k)+'.pdf')
                    plt.close()

                
