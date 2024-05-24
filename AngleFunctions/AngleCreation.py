# Python script for computing rodrigues angles and comparing different angle measures

# Routine: main.py
# Author: P.D. Ledger
# Date: 18/1/2024

# include a few standard libaries
from time import time
import numpy as np
import os
import sys
from matplotlib import pyplot as plt

from EigenValueDecomposition import *
from MinMaxthetafromQRQI import *
from SortEigenValues import *
from AnglesSortedQRQI import *
from SplitTensor import *
from Fmeasure import *
from Commeasure import *



#   Main script pass arguments of
#   directory = Directory in which Tensors.csv, N0.csv and Frequencies.csv are stored
#   MaxOmega = Maximum frequency to be considered
def AngleCreation(TensorArray,N0,Frequencies,Figures="Off",FullRom="Rom"):
   
    N=len(Frequencies)
    # N0 is generated from scaling and possible interpolation. Force symmetry
    N0 = (N0 + np.transpose(N0))/2.
   
    Rstore,Istore,Rtildestore, N0store = SplitTensor(TensorArray,Frequencies,N0)

    # Determine eigenvalue decompositions of N0, R, I, Rtilde (no sorting applied), and their multiplicities
    MultRstore, MultIstore, MultRtildestore, MultN0store, URstore, UIstore, URtildestore, UN0store, QRstore, QIstore, QRtildestore, QN0store = EigenValueDecomposition(N0,TensorArray,Frequencies)

    # Indicate which cases to consider
    RImeasure=False
    RtildeImeasure=True
    N0Imeasure=False
    N0Rmeasure=False
    N0Rtildemeasure=False

    # Determine the maximal and minimal angles from QR and QI also output d_F metric for these orderings
    FixEvecs="Yes"#"Yes"
    #MinAnglestoreRI, MaxAnglestoreRI, dFMinAnglestoreRI, dFMaxAnglestoreRI,SortedURstore, SortedUIstore, SortedQRstore, SortedQIstore, SortedKstore = MinMaxthetafromQRQI(Frequencies,QRstore,QIstore,URstore, UIstore,MultRstore,MultIstore,FixEvecs)
    SortedMultRstore=[]
    SortedMultIstore=[]

    if RImeasure == True:
        # Sort eigenvalues (and eigenvectors) so that || Lambda_R - Lambda_I || is maximal
        sorteigenvalues="MaxDifference"
        SortedMultRstore, SortedMultIstore, SortedURstore, SortedUIstore, SortedQRstore, SortedQIstore, SortedKstore = SortEigenValues(MultRstore, MultIstore, URstore, UIstore, QRstore, QIstore, Frequencies, sorteigenvalues, Rstore, Istore)

        #Obtain f meauses (approx and exact constant)
        AnglestoreRIfmeasfullconstsortedmaxdiff, AnglestoreRIfmeasapprxconstsortedmaxdiff_min,AnglestoreRIfmeasapprxconstsortedmaxdiff_max, RIfmeasapprx_den_const_max = Fmeasure(sorteigenvalues,SortedURstore, SortedUIstore, SortedQRstore, SortedQIstore, SortedKstore, Rstore,Istore, Frequencies)

        #Obtain Com-meauses (approx and exact constant)
        AnglestoreRIcommeasfullconstsortedmaxdiff, AnglestoreRIcommeasapprxconstsortedmaxdiff_min,AnglestoreRIcommeasapprxconstsortedmaxdiff_max, RIcommeapprx_den_const_max = Commeasure(sorteigenvalues,SortedURstore, SortedUIstore, SortedQRstore, SortedQIstore, SortedKstore, Rstore,Istore, Frequencies)

        # Sort eigenvalues (and eigenvectors) so that || Lambda_R - Lambda_I || is minimal
        sorteigenvalues="MinDifference"
        SortedMultRstore, SortedMultIstore, SortedURstore, SortedUIstore, SortedQRstore, SortedQIstore, SortedKstore = SortEigenValues(MultRstore, MultIstore, URstore, UIstore, QRstore, QIstore, Frequencies, sorteigenvalues, Rstore, Istore)

        #Obtain f meauses (approx and exact constant)
        AnglestoreRIfmeasfullconstsortedmindiff, AnglestoreRIfmeasapprxconstsortedmindiff_min, AnglestoreRIfmeasapprxconstsortedmindiff_max, RIfmeasapprx_den_const_min = Fmeasure(sorteigenvalues,SortedURstore, SortedUIstore, SortedQRstore, SortedQIstore, SortedKstore, Rstore,Istore, Frequencies)

        #Obtain Com-meauses (approx and exact constant)
        AnglestoreRIcommeasfullconstsortedmindiff, AnglestoreRIcommeasapprxconstsortedmindiff_min,AnglestoreRIcommeasapprxconstsortedmindiff_max, RIcommeapprx_den_const_min = Commeasure(sorteigenvalues,SortedURstore, SortedUIstore, SortedQRstore, SortedQIstore, SortedKstore, Rstore,Istore, Frequencies)

        kthetavec=np.zeros((N,3))
        if Figures=="On":
            fig=plt.figure()
            plt.semilogx(Frequencies,np.fmin(AnglestoreRIfmeasapprxconstsortedmaxdiff_min,AnglestoreRIfmeasapprxconstsortedmindiff_min),'x',label=r'Approx $d_R({\cal R},{\cal I})$ from  $d_E({\cal R},{\cal I})$ ')

            plt.semilogx(Frequencies,np.fmin(AnglestoreRIcommeasapprxconstsortedmaxdiff_min,AnglestoreRIcommeasapprxconstsortedmindiff_min),'x',label=r'Approx $d_R({\cal R},{\cal I})$ from  $d_C({\cal R},{\cal I})$ ')

            plt.xlabel(r'$\omega$ [rad/s]')
            plt.ylabel(r'$\theta$ [rad]')
            plt.legend()
            plt.show()


        thetaRIdE=np.fmin(AnglestoreRIfmeasapprxconstsortedmaxdiff_min,AnglestoreRIfmeasapprxconstsortedmindiff_min)
        thetaRIdC=np.fmin(AnglestoreRIcommeasapprxconstsortedmaxdiff_min,AnglestoreRIcommeasapprxconstsortedmindiff_min)
        RIResults={"Frequencies": Frequencies,"thetaRIdE":thetaRIdE,"thetaRIdC":thetaRIdC}
    else:
        RIResults=[]


    if RtildeImeasure==True:
        # Sort eigenvalues (and eigenvectors) so that || Lambda_Rtilde - Lambda_I || is maximal
        sorteigenvalues="MaxDifference"
        SortedMultRtildestore, SortedMultIstore, SortedURtildestore, SortedUIstore, SortedQRtildestore, SortedQIstore, SortedKstore = SortEigenValues(MultRtildestore, MultIstore, URtildestore, UIstore, QRtildestore, QIstore, Frequencies, sorteigenvalues, Rtildestore, Istore)

        #print('Computing F measure Tilde')
        AnglestoreRtildeIfmeasfullconstsortedmaxdiff, AnglestoreRtildeIfmeasapprxconstsortedmaxdiff_min, AnglestoreRtildeIfmeasapprxconstsortedmaxdiff_max, RtildeIfmeasapprx_den_const_max = Fmeasure(sorteigenvalues,SortedURtildestore, SortedUIstore, SortedQRtildestore, SortedQIstore, SortedKstore, Rtildestore,Istore, Frequencies)

        #Obtain Com meauses (approx and exact constant)
        AnglestoreRtildeIcommeasfullconstsortedmaxdiff, AnglestoreRtildeIcommeasapprxconstsortedmaxdiff_min, AnglestoreRtildeIcommeasapprxconstsortedmaxdiff_max, RtildeIcommeapprx_den_const_max = Commeasure(sorteigenvalues,SortedURtildestore, SortedUIstore, SortedQRtildestore, SortedQIstore, SortedKstore, Rtildestore,Istore, Frequencies)


        # Sort eigenvalues (and eigenvectors) so that || Lambda_R - Lambda_I || is minimal
        sorteigenvalues="MinDifference"
        SortedMultRtildestore, SortedMultIstore, SortedURtildestore, SortedUIstore, SortedQRtildestore, SortedQIstore, SortedKstore = SortEigenValues(MultRtildestore, MultIstore, URtildestore, UIstore, QRtildestore, QIstore, Frequencies, sorteigenvalues, Rtildestore, Istore)

        #Obtain f meauses (approx and exact constant)
        AnglestoreRtildeIfmeasfullconstsortedmindiff, AnglestoreRtildeIfmeasapprxconstsortedmindiff_min,AnglestoreRtildeIfmeasapprxconstsortedmindiff_max, RtildeIfmeasapprx_den_const_min = Fmeasure(sorteigenvalues,SortedURtildestore, SortedUIstore, SortedQRtildestore, SortedQIstore, SortedKstore, Rtildestore,Istore, Frequencies)

        #Obtain com meauses (approx and exact constant)
        AnglestoreRtildeIcommeasfullconstsortedmindiff, AnglestoreRtildeIcommeasapprxconstsortedmindiff_min,AnglestoreRtildeIcommeasapprxconstsortedmindiff_max, RtildeIcommeapprx_den_const_min = Commeasure(sorteigenvalues,SortedURtildestore, SortedUIstore, SortedQRtildestore, SortedQIstore, SortedKstore, Rtildestore,Istore, Frequencies)

        kthetavec=np.zeros((N,3))

        if Figures=="On":
            fig=plt.figure()
            plt.semilogx(Frequencies,np.fmin(AnglestoreRtildeIfmeasapprxconstsortedmaxdiff_min,AnglestoreRtildeIfmeasapprxconstsortedmindiff_min),'x',label=r'Approx $d_R(\tilde{\cal R},{\cal I})$ from  $d_E(\tilde{\cal R},{\cal I})$ ')
            plt.semilogx(Frequencies,np.fmin(AnglestoreRtildeIcommeasapprxconstsortedmaxdiff_min,AnglestoreRtildeIcommeasapprxconstsortedmindiff_min),'x',label=r'Approx $d_R(\tilde{\cal R},{\cal I})$ from  $d_C(\tilde{\cal R},{\cal I})$ ')
            plt.xlabel(r'$\omega$ [rad/s]')
            plt.ylabel(r'$\theta$ [rad]')
            plt.legend()
            plt.show()

        thetaRtildeIdE=np.fmin(AnglestoreRtildeIfmeasapprxconstsortedmaxdiff_min,AnglestoreRtildeIfmeasapprxconstsortedmindiff_min)
        thetaRtildeIdC=np.fmin(AnglestoreRtildeIcommeasapprxconstsortedmaxdiff_min,AnglestoreRtildeIcommeasapprxconstsortedmindiff_min)
        RtildeIResults={"Frequencies": Frequencies,"thetaRtildeIdE":thetaRtildeIdE,"thetaRtildeIdC":thetaRtildeIdC}
    else:
        RtildeIResults=[]


    if N0Imeasure==True:
        # Sort eigenvalues (and eigenvectors) so that || Lambda_Rtilde - Lambda_I || is maximal
        sorteigenvalues="MaxDifference"
        SortedMultN0store, SortedMultIstore, SortedUN0store, SortedUIstore, SortedQN0store, SortedQIstore, SortedKstore = SortEigenValues(MultN0store, MultIstore, UN0store, UIstore, QN0store, QIstore, Frequencies, sorteigenvalues, N0store, Istore)

        #Obtain f meauses (approx and exact constant)
        AnglestoreN0Ifmeasfullconstsortedmaxdiff, AnglestoreN0Ifmeasapprxconstsortedmaxdiff_min, AnglestoreN0Ifmeasapprxconstsortedmaxdiff_max, N0Ifmeasapprx_den_const_max = Fmeasure(sorteigenvalues,SortedUN0store, SortedUIstore, SortedQN0store, SortedQIstore, SortedKstore, N0store,Istore, Frequencies)

        #Obtain Com meauses (approx and exact constant)
        AnglestoreN0Icommeasfullconstsortedmaxdiff, AnglestoreN0Icommeasapprxconstsortedmaxdiff_min, AnglestoreN0Icommeasapprxconstsortedmaxdiff_max, N0Icommeapprx_den_const_max = Commeasure(sorteigenvalues,SortedUN0store, SortedUIstore, SortedQN0store, SortedQIstore, SortedKstore, N0store,Istore, Frequencies)


        # Sort eigenvalues (and eigenvectors) so that || Lambda_R - Lambda_I || is minimal
        sorteigenvalues="MinDifference"
        SortedMultN0store, SortedMultIstore, SortedUN0store, SortedUIstore, SortedQN0store, SortedQIstore, SortedKstore = SortEigenValues(MultN0store, MultIstore, UN0store, UIstore, QN0store, QIstore, Frequencies, sorteigenvalues, N0store, Istore)

        #Obtain f meauses (approx and exact constant)
        AnglestoreN0Ifmeasfullconstsortedmindiff, AnglestoreN0Ifmeasapprxconstsortedmindiff_min,AnglestoreN0Ifmeasapprxconstsortedmindiff_max, N0Ifmeasapprx_den_const_min = Fmeasure(sorteigenvalues,SortedUN0store, SortedUIstore, SortedQN0store, SortedQIstore, SortedKstore, N0store,Istore, Frequencies)

        #Obtain com meauses (approx and exact constant)
        AnglestoreN0Icommeasfullconstsortedmindiff, AnglestoreN0Icommeasapprxconstsortedmindiff_min,AnglestoreN0Icommeasapprxconstsortedmindiff_max, N0Icommeapprx_den_const_min= Commeasure(sorteigenvalues,SortedUN0store, SortedUIstore, SortedQN0store, SortedQIstore, SortedKstore, N0store,Istore, Frequencies)

        if Figures=="On":
            fig=plt.figure()
            plt.semilogx(Frequencies,np.fmin(AnglestoreN0Ifmeasapprxconstsortedmaxdiff_min,AnglestoreN0Ifmeasapprxconstsortedmindiff_min),'x',label=r'Approx $d_R( {\cal N}^{(0)},{\cal I})$ from  $d_E( {\cal N}^{(0)},{\cal I})$ ')
            plt.semilogx(Frequencies,np.fmin(AnglestoreN0Icommeasapprxconstsortedmaxdiff_min,AnglestoreN0Icommeasapprxconstsortedmindiff_min),'x',label=r'Approx $d_R( {\cal N}^{(0)},{\cal I})$ from  $d_C( {\cal N}^{(0)},{\cal I})$ ')
            plt.xlabel(r'$\omega$ [rad/s]')
            plt.ylabel(r'$\theta$ [rad]')
            plt.legend()
            plt.show()

        thetaN0IdE=np.fmin(AnglestoreN0Ifmeasapprxconstsortedmaxdiff_min,AnglestoreN0Ifmeasapprxconstsortedmindiff_min)
        thetaN0IdC=np.fmin(AnglestoreN0Icommeasapprxconstsortedmaxdiff_min,AnglestoreN0Icommeasapprxconstsortedmindiff_min)
        N0IResults={"Frequencies": Frequencies,"thetaN0IdE":thetaN0IdE,"thetaN0IdC":thetaN0IdC}
    else:
        N0IResults=[]



########################## N0  and # R
    if N0Rmeasure==True:

        # Sort eigenvalues (and eigenvectors) so that || Lambda_Rtilde - Lambda_I || is maximal
        sorteigenvalues="MaxDifference"
        SortedMultN0store, SortedMultRstore, SortedUN0store, SortedURstore, SortedQN0store, SortedQRstore, SortedKstore = SortEigenValues(MultN0store, MultRstore, UN0store, URstore, QN0store, QRstore, Frequencies, sorteigenvalues, N0store, Rstore)

        #Obtain f meauses (approx and exact constant)
        AnglestoreN0Rfmeasfullconstsortedmaxdiff, AnglestoreN0Rfmeasapprxconstsortedmaxdiff_min, AnglestoreN0Rfmeasapprxconstsortedmaxdiff_max, N0Rfmeasapprx_den_const_max = Fmeasure(sorteigenvalues,SortedUN0store, SortedURstore, SortedQN0store, SortedQRstore, SortedKstore, N0store,Rstore, Frequencies)

        #Obtain Com meauses (approx and exact constant)
        AnglestoreN0Rcommeasfullconstsortedmaxdiff, AnglestoreN0Rcommeasapprxconstsortedmaxdiff_min, AnglestoreN0Rcommeasapprxconstsortedmaxdiff_max, N0Rcommeapprx_den_const_max = Commeasure(sorteigenvalues,SortedUN0store, SortedURstore, SortedQN0store, SortedQRstore, SortedKstore, N0store,Rstore, Frequencies)


        # Sort eigenvalues (and eigenvectors) so that || Lambda_R - Lambda_I || is minimal
        sorteigenvalues="MinDifference"
        SortedMultN0store, SortedMultRstore, SortedUN0store, SortedURstore, SortedQN0store, SortedQRstore, SortedKstore = SortEigenValues(MultN0store, MultRstore, UN0store, URstore, QN0store, QRstore, Frequencies, sorteigenvalues, N0store, Rstore)

        #Obtain f meauses (approx and exact constant)
        AnglestoreN0Rfmeasfullconstsortedmindiff, AnglestoreN0Rfmeasapprxconstsortedmindiff_min,AnglestoreN0Rfmeasapprxconstsortedmindiff_max, N0Rfmeasapprx_den_const_min = Fmeasure(sorteigenvalues,SortedUN0store, SortedURstore, SortedQN0store, SortedQRstore, SortedKstore, N0store,Rstore, Frequencies)

        #Obtain com meauses (approx and exact constant)
        AnglestoreN0Rcommeasfullconstsortedmindiff, AnglestoreN0Rcommeasapprxconstsortedmindiff_min,AnglestoreN0Rcommeasapprxconstsortedmindiff_max, N0Rcommeapprx_den_const_min= Commeasure(sorteigenvalues,SortedUN0store, SortedURstore, SortedQN0store, SortedQRstore, SortedKstore, N0store,Rstore, Frequencies)

        kthetavec=np.zeros((N,3))

        if Figures=="On":
            fig=plt.figure()
            plt.semilogx(Frequencies,np.fmin(AnglestoreN0Rfmeasapprxconstsortedmaxdiff_min,AnglestoreN0Rfmeasapprxconstsortedmindiff_min),'x',label=r'Approx $d_R( {\cal N}^{(0)},{\cal R})$ from  $d_E( {\cal N}^{(0)},{\cal R})$ ')
            plt.semilogx(Frequencies,np.fmin(AnglestoreN0Rcommeasapprxconstsortedmaxdiff_min,AnglestoreN0Rcommeasapprxconstsortedmindiff_min),'x',label=r'Approx $d_R( {\cal N}^{(0)},{\cal R})$ from  $d_C( {\cal N}^{(0)},{\cal R})$ ')
            plt.xlabel(r'$\omega$ [rad/s]')
            plt.ylabel(r'$\theta$ [rad]')
            plt.legend()
            plt.show()

        thetaN0RdE=np.fmin(AnglestoreN0Rfmeasapprxconstsortedmaxdiff_min,AnglestoreN0Rfmeasapprxconstsortedmindiff_min)
        thetaN0RdC=np.fmin(AnglestoreN0Rcommeasapprxconstsortedmaxdiff_min,AnglestoreN0Rcommeasapprxconstsortedmindiff_min)
        N0RResults={"Frequencies": Frequencies,"thetaN0RdE":thetaN0RdE,"thetaN0RdC":thetaN0RdC}
    else:
        N0RResults=[]


######################### N0  and # Rtilde
    if N0Rtildemeasure==True:

        # Sort eigenvalues (and eigenvectors) so that || Lambda_Rtilde - Lambda_I || is maximal
        sorteigenvalues="MaxDifference"
        # Obtain angles for this sorted min-max combination
        AnglestoreN0Rtildesortedmaxdiff = AnglesSortedQRQI(SortedQN0store,SortedQRtildestore,Frequencies)

        #Obtain f meauses (approx and exact constant)
        AnglestoreN0Rtildefmeasfullconstsortedmaxdiff, AnglestoreN0Rtildefmeasapprxconstsortedmaxdiff_min,\
        AnglestoreN0Rtildefmeasapprxconstsortedmaxdiff_max, N0Rtildefmeasapprx_den_const_max = Fmeasure(sorteigenvalues,SortedUN0store, SortedURtildestore,
                                                                                                        SortedQN0store, SortedQRtildestore, SortedKstore,
                                                                                                        N0store,Rtildestore, Frequencies)

        #Obtain Com meauses (approx and exact constant)
        AnglestoreN0Rtildecommeasfullconstsortedmaxdiff, AnglestoreN0Rtildecommeasapprxconstsortedmaxdiff_min, AnglestoreN0Rtildecommeasapprxconstsortedmaxdiff_max, N0Rtildecommeapprx_den_const_max = Commeasure(sorteigenvalues,SortedUN0store, SortedURtildestore, SortedQN0store, SortedQRtildestore, SortedKstore, N0store,Rtildestore, Frequencies)


        # Sort eigenvalues (and eigenvectors) so that || Lambda_R - Lambda_I || is minimal
        sorteigenvalues="MinDifference"
        SortedMultN0store, SortedMultRtildestore, SortedUN0store, SortedURtildestore, SortedQN0store, SortedQRtildestore, SortedKstore = SortEigenValues(MultN0store, MultRtildestore, UN0store, URtildestore, QN0store, QRtildestore, Frequencies, sorteigenvalues, N0store, Rtildestore)

        #Obtain f meauses (approx and exact constant)
        AnglestoreN0Rtildefmeasfullconstsortedmindiff, AnglestoreN0Rtildefmeasapprxconstsortedmindiff_min,\
        AnglestoreN0Rtildefmeasapprxconstsortedmindiff_max, N0Rtildefmeasapprx_den_const_min = Fmeasure(sorteigenvalues,SortedUN0store, SortedURtildestore,
                                                                                                        SortedQN0store, SortedQRtildestore, SortedKstore,
                                                                                                        N0store,Rtildestore, Frequencies)

        #Obtain com meauses (approx and exact constant)
        AnglestoreN0Rtildecommeasfullconstsortedmindiff, AnglestoreN0Rtildecommeasapprxconstsortedmindiff_min,AnglestoreN0Rtildecommeasapprxconstsortedmindiff_max, N0Rtildecommeapprx_den_const_min= Commeasure(sorteigenvalues,SortedUN0store, SortedURtildestore, SortedQN0store, SortedQRtildestore, SortedKstore, N0store,Rtildestore, Frequencies)

        kthetavec=np.zeros((N,3))

        if Figures=="On":
            fig=plt.figure()
            plt.semilogx(Frequencies,np.fmin(AnglestoreN0Rtildefmeasapprxconstsortedmaxdiff_min,AnglestoreN0Rtildefmeasapprxconstsortedmindiff_min),'x',label=r'Approx $d_R( {\cal N}^{(0)},\tilde{\cal R})$ from  $d_E( {\cal N}^{(0)},\tilde{\cal R})$ ')
            plt.semilogx(Frequencies,np.fmin(AnglestoreN0Rtildecommeasapprxconstsortedmaxdiff_min,AnglestoreN0Rtildecommeasapprxconstsortedmindiff_min),'x',label=r'Approx $d_R( {\cal N}^{(0)},\tilde{\cal R})$ from  $d_C( {\cal N}^{(0)},\tilde{\cal R})$ ')
            plt.xlabel(r'$\omega$ [rad/s]')
            plt.ylabel(r'$\theta$ [rad]')
            plt.legend()
            plt.show()


        thetaN0RtildedE=np.fmin(AnglestoreN0Rtildefmeasapprxconstsortedmaxdiff_min,AnglestoreN0Rtildefmeasapprxconstsortedmindiff_min)
        thetaN0RtildedC=np.fmin(AnglestoreN0Rtildecommeasapprxconstsortedmaxdiff_min,AnglestoreN0Rtildecommeasapprxconstsortedmindiff_min)
        N0RtildeResults={"Frequencies": Frequencies,"thetaN0RtildedE":thetaN0RtildedE,"thetaN0RtildedC":thetaN0RtildedC}
    #
    else:
        N0RtildeResults=[]




    #return RIResults,RtildeIResults,N0IResults,N0RResults,N0RtildeResults
    return RtildeIResults["thetaRtildeIdE"]
