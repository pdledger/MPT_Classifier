import os
import sys
from sys import exit
import math
import csv
import joblib
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
#from sklearn import model_selection, svm, tree, ensemble
#from sklearn.linear_model import LogisticRegression
#from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score
from scipy.stats import sem, t
import pandas as pd
import seaborn as sns
import time
from scipy.stats import gaussian_kde

def Check_KL_divergence(X,Y):

    N,D=np.shape(X)
    # N is the number of data points
    # D is the number of Features
    # Y provides the class label for each data point
    Number_Of_Classes=int(np.max(Y))+1
    print(Number_Of_Classes,int(np.min(Y)),D,N)

    # Order the data ready for obtaining the mean and covairance
    Data=np.zeros((Number_Of_Classes,N,D))
    cnt=np.zeros(Number_Of_Classes, dtype=int) # how many times a set of features is tagged for each class
    for n in range(N):
        for f in range(D):
            #print(int(Y[n]),cnt[int(Y[n])],f)
            Data[int(Y[n]),cnt[int(Y[n])],f]=X[n,f]
        cnt[int(Y[n])]+=1
    # Data will be order number of classes x number of datasets for this class x features

    # Find the mean vectors
    mu=np.zeros((Number_Of_Classes,D))
    for nclass in range(Number_Of_Classes):
        data_for_this_class=np.zeros((cnt[nclass],D))
        data_for_this_class[:,:]=np.copy(Data[nclass,0:cnt[nclass],:]) # number of datasets x number of features
        mu[nclass,:]=np.mean(data_for_this_class,axis=0)
        print(mu[nclass,:])

    ## Find the covairance matrices
    covairance=np.zeros((Number_Of_Classes,D,D))
    invcovairance=np.zeros((Number_Of_Classes,D,D))

    for nclass in range(Number_Of_Classes):
        data_for_this_class=np.zeros((cnt[nclass],D))
        data_for_this_class[:,:]=np.copy(Data[nclass,0:cnt[nclass],:]) # this is a number of data points  x features
        kernel_p=gaussian_kde(data_for_this_class.T)
        covairance[nclass,:,:]=kernel_p.covariance/kernel_p.factor**2
        invcovairance[nclass,:,:]=kernel_p.inv_cov*kernel_p.factor**2
        #covairance[nclass,:,:]=np.cov(data_for_this_class.T) #nb this expects features x number of data points
        #print(nclass,np.cov(data_for_this_class.T))



    # Compute the metric
    F=0.
    for nclass in range(Number_Of_Classes):
        for mclass in range(Number_Of_Classes):
            if nclass != mclass :
                F+= kl_div_gauss(mu[nclass,:],mu[mclass,:],covairance[nclass,:,:],invcovairance[nclass,:,:],covairance[mclass,:,:],invcovairance[mclass,:,:],D)
                #print(kl_div_gauss(mu[nclass,:],mu[mclass,:],covairance[nclass,:,:],covairance[mclass,:,:],D))
    print(F)

    return 0

def kl_div_gauss(mui,muj,sigmai,invsigmai,sigmaj,invsigmaj,D):
    dsigmai=np.linalg.det(sigmai)
    dsigmaj=np.linalg.det(sigmaj)
    #invsigmaj=np.linalg.inv(sigmaj)
    #print(dsigmai,dsigmaj,invsigmaj)

    return 1/2.*(mui-muj).T@(invsigmaj@(mui-muj)) + 1/2.*(np.trace(invsigmaj@sigmai)-D+np.log(np.abs(dsigmaj/dsigmai)))
