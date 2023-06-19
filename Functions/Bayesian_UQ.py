import os
import sys
from sys import exit
import math
import numpy as np

def Bayesian_UQ(Number_Of_Classes,UQPL,ActualPL,ProbabilitiesPL,ProbabilitiesUpPL,ProbabilitiesLowPL):
   # Looks like the median is stored as the mean.
    Object_Confidence_Mean = np.zeros([Number_Of_Classes,Number_Of_Classes])
    Object_Confidence_Confidence = np.zeros([Number_Of_Classes,Number_Of_Classes,4])
    Object_Percentiles = np.zeros([Number_Of_Classes,Number_Of_Classes,101])
    Object_UQ_minval_up = np.zeros([Number_Of_Classes,Number_Of_Classes])
    Object_UQ_minval_val = np.zeros([Number_Of_Classes,Number_Of_Classes])
    Object_UQ_minval_low = np.zeros([Number_Of_Classes,Number_Of_Classes])

    Object_UQ_maxval_up = np.zeros([Number_Of_Classes,Number_Of_Classes])
    Object_UQ_maxval_val = np.zeros([Number_Of_Classes,Number_Of_Classes])
    Object_UQ_maxval_low = np.zeros([Number_Of_Classes,Number_Of_Classes])

    maxuq = np.max(np.max(UQPL))
    bins = np.logspace(-4,np.log10(maxuq),40)

    Hist = np.zeros([Number_Of_Classes,40-1])
    Bin_edges = np.zeros([Number_Of_Classes,40])

    for i in range(Number_Of_Classes):
#                    temp_probs = np.array([prob for j,prob in enumerate(Probabilities_appended) if Truth_list[j]==i])
        row = len(ActualPL)

        temp_probs = np.empty(Number_Of_Classes)
        count = 0
        uqmin = 1.e100
        uqmax = 0.
        for j in range(row):
            # If the ith class considered is the same as the true class of the jth output
            if i == int(ActualPL[j]):
                if count == 0:
                    temp_probs = ProbabilitiesPL[j,:]
                    UQh = np.max(UQPL[j,:])
                    count = count + 1
                else:
                    temp_probs = np.vstack((temp_probs, ProbabilitiesPL[j,:]))
                    UQh = np.append(UQh, np.max(UQPL[j,:]))
                    count = count + 1
                # This will keep min and max uncertainity if the ith class is the same as true class of the jth output
                if np.max(UQPL[j,:]) > uqmax:
                    Object_UQ_maxval_up[i,:] =  ProbabilitiesUpPL[j,:]
                    Object_UQ_maxval_val[i,:] = ProbabilitiesPL[j,:]
                    Object_UQ_maxval_low[i,:] = ProbabilitiesLowPL[j,:]
                    uqmax = np.max(UQPL[j,:])
                if np.min(UQPL[j,:]) < uqmin:
                    Object_UQ_minval_up[i,:] =  ProbabilitiesUpPL[j,:]
                    Object_UQ_minval_val[i,:] = ProbabilitiesPL[j,:]
                    Object_UQ_minval_low[i,:] = ProbabilitiesLowPL[j,:]
                    uqmin = np.min(UQPL[j,:])

        #print(temp_probs,np.shape(temp_probs))
        #time.sleep(10)

        # Determine histogram
        hist,bin_edges = np.histogram(UQh,bins=bins)
        # Store relative frequency
        Hist[i,:] = hist/float(count)
        Bin_edges[i,:] = bin_edges

        Sorted_posteriors = np.zeros(np.shape(temp_probs))
        for j in range(Number_Of_Classes):
#                        Sorted_posteriors[:,j] = np.sort(temp_probs[:,j])
#                        Object_Confidence_Confidence[i,j,0] = Sorted_posteriors[math.ceil(len(Sorted_posteriors)/40),j]
#                        Object_Confidence_Confidence[i,j,1] = Sorted_posteriors[math.ceil(len(Sorted_posteriors)/4),j]
#                        # This looks like the median is stored as the mean
#                        Object_Confidence_Mean[i,j] = Sorted_posteriors[math.ceil(len(Sorted_posteriors)/2),j]
#                        Object_Confidence_Confidence[i,j,2] = Sorted_posteriors[math.floor(len(Sorted_posteriors)*3/4),j]
#                        Object_Confidence_Confidence[i,j,3] = Sorted_posteriors[math.floor(len(Sorted_posteriors)*39/40),j]
#                        for l in range(100):
#
#                            try:
#                                Object_Percentiles[i,j,l] = Sorted_posteriors[math.ceil(len(Sorted_posteriors)*l/100),j]
#                            except:
#                                pass
#                        Object_Percentiles[i,j,100] = Sorted_posteriors[-1,j]
            Sorted_posteriors = np.sort(temp_probs[:,j])
            Object_Confidence_Confidence[i,j,0] = Sorted_posteriors[math.ceil(len(Sorted_posteriors)/40)]
            Object_Confidence_Confidence[i,j,1] = Sorted_posteriors[math.ceil(len(Sorted_posteriors)/4)]
            # This looks like the median is stored as the mean
            Object_Confidence_Mean[i,j] = Sorted_posteriors[math.ceil(len(Sorted_posteriors)/2)]
            Object_Confidence_Confidence[i,j,2] = Sorted_posteriors[math.floor(len(Sorted_posteriors)*3/4)]
            Object_Confidence_Confidence[i,j,3] = Sorted_posteriors[math.floor(len(Sorted_posteriors)*39/40)]
            for l in range(100):

                try:
                    Object_Percentiles[i,j,l] = Sorted_posteriors[math.ceil(len(Sorted_posteriors)*l/100)]
                except:
                    pass
            Object_Percentiles[i,j,100] = Sorted_posteriors[-1]
    return Object_Confidence_Confidence,Object_Percentiles,Object_Confidence_Mean,Object_UQ_minval_val, Object_UQ_minval_low,Object_UQ_minval_up,Object_UQ_maxval_val,Object_UQ_maxval_low,Object_UQ_maxval_up,Bin_edges,Hist, Sorted_posteriors

