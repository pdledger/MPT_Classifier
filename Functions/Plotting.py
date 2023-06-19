import os
import sys
from sys import exit
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns


def Plotting(Model,Probabalistic_Classifiers,Bayesian_Classifiers,Object_Confidence_Confidence,Object_Percentiles,DataSet_Name,Savename,Object_Confidence_Mean,Number_Of_Classes,PYCOL,Object_UQ_minval_val,Object_UQ_minval_low,Object_UQ_minval_up,Object_UQ_maxval_val,Object_UQ_maxval_low,Object_UQ_maxval_up,Bin_edges,Hist,Testing_noise,reordered_names):

    if (Model in Probabalistic_Classifiers) or (Model in Bayesian_Classifiers ): #or ('MLP' in Model):
        Object_Confidence_Confidence_saving = Object_Confidence_Confidence.reshape(Object_Confidence_Confidence.shape[0], -1)
        Object_Percentiles_saving = Object_Percentiles.reshape(Object_Percentiles.shape[0], -1)
    if type(Testing_noise) == bool:
        # Again looks like median saved as mean
        np.savetxt('Results/'+DataSet_Name+'/Noiseless/'+Model+'/'+Savename+'/Posteriors_Mat.csv',Object_Confidence_Mean,delimiter=',')
        np.savetxt('Results/'+DataSet_Name+'/Noiseless/'+Model+'/'+Savename+'/Confidence_Mat.csv',Object_Confidence_Confidence_saving,delimiter=',')
        np.savetxt('Results/'+DataSet_Name+'/Noiseless/'+Model+'/'+Savename+'/Percentiles_Mat.csv',Object_Percentiles_saving,delimiter=',')
    else:
        # Again looks like median saved as mean
        np.savetxt('Results/'+DataSet_Name+'/Noise_'+str(Testing_noise)+'/'+Model+'/'+Savename+'/Posteriors_Mat.csv',Object_Confidence_Mean,delimiter=',')
        np.savetxt('Results/'+DataSet_Name+'/Noise_'+str(Testing_noise)+'/'+Model+'/'+Savename+'/Confidence_Mat.csv',Object_Confidence_Confidence_saving,delimiter=',')
        np.savetxt('Results/'+DataSet_Name+'/Noise_'+str(Testing_noise)+'/'+Model+'/'+Savename+'/Percentiles_Mat.csv',Object_Percentiles_saving,delimiter=',')

#               Plot out bar graphs for median and 2.5, 97.5 % percentiles
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


#               Plot out bar graphs for median and Q1, Q3 quartiles
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

#               Plot out bar graphs showing output for min uq value
    labels = ['Probability','UQ min']
    for i in range(Number_Of_Classes):
        fig, ax = plt.subplots()
        Bars = ax.bar(np.arange(Number_Of_Classes), Object_UQ_minval_val[i,:],color=[PYCOL[3] if np.argmax(Object_UQ_minval_val[i,:])==j else PYCOL[0] for j in range(Number_Of_Classes)], align='center', alpha=0.5, ecolor='black', capsize=10,label='Probability')

        # including upper limits
        for j in range(Number_Of_Classes):
            ax.plot([j,j],[Object_UQ_minval_low[i,j],Object_UQ_minval_up[i,j]],linestyle='-', marker='_',markersize=16,color='k')

        Legend_elements = [Patch(facecolor=PYCOL[0],alpha=0.6,label=r'Posterior probability $p(C_k|$data)'),Patch(facecolor=PYCOL[3],alpha=0.6,label=r'MAP estimate max$(p(C_k|$data))'),plt.Line2D([0],[0],color='k', marker='_', markersize=16 ,linestyle=' ', label='Min UQ')]
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
            plt.savefig('Results/'+DataSet_Name+'/Noiseless/'+Model+'/'+Savename+'/figure'+str(i+2*Number_Of_Classes+1)+'.pdf')
        else:
            plt.savefig('Results/'+DataSet_Name+'/Noise_'+str(Testing_noise)+'/'+Model+'/'+Savename+'/figure'+str(i+2*Number_Of_Classes+1)+'.pdf')
        plt.close()


#               Plot out bar graphs showing output for max uq value
    labels = ['Probability','UQ max']
    for i in range(Number_Of_Classes):
        fig, ax = plt.subplots()
        Bars = ax.bar(np.arange(Number_Of_Classes), Object_UQ_maxval_val[i,:],color=[PYCOL[3] if np.argmax(Object_UQ_maxval_val[i,:])==j else PYCOL[0] for j in range(Number_Of_Classes)], align='center', alpha=0.5, ecolor='black', capsize=10,label='Probability')

        # including upper limits
        for j in range(Number_Of_Classes):
            ax.plot([j,j],[ Object_UQ_maxval_low[i,j], Object_UQ_maxval_up[i,j] ],linestyle='-', marker='_',markersize=16,color='k')

        Legend_elements = [Patch(facecolor=PYCOL[0],alpha=0.6,label=r'Posterior probability $p(C_k|$data)'),Patch(facecolor=PYCOL[3],alpha=0.6,label=r'MAP estimate max$(p(C_k|$data))'),plt.Line2D([0],[0],color='k', marker='_', markersize=16 ,linestyle=' ', label='Max UQ')]
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
            plt.savefig('Results/'+DataSet_Name+'/Noiseless/'+Model+'/'+Savename+'/figure'+str(i+3*Number_Of_Classes+1)+'.pdf')
        else:
            plt.savefig('Results/'+DataSet_Name+'/Noise_'+str(Testing_noise)+'/'+Model+'/'+Savename+'/figure'+str(i+3*Number_Of_Classes+1)+'.pdf')
        plt.close()

    # Plot out the histograms of UQ for different Classes
    fig, ax = plt.subplots()
    legendtxt=[]
    for i in range(Number_Of_Classes):
        ax.semilogx((Bin_edges[i,:-1]+Bin_edges[i,1:])/2.,Hist[i,:],label='C_'+str(i+1))
        #print(legendtxt)
    plt.xlabel('UQ measure')
    plt.ylabel('Relative Frequency')
    plt.legend()
    if type(Testing_noise) == bool:
        plt.savefig('Results/'+DataSet_Name+'/Noiseless/'+Model+'/'+Savename+'/figure'+str(4*Number_Of_Classes+1)+'.pdf')
    else:
        plt.savefig('Results/'+DataSet_Name+'/Noise_'+str(Testing_noise)+'/'+Model+'/'+Savename+'/figure'+str(4*Number_Of_Classes+1)+'.pdf')
    plt.close()
