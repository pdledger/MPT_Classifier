########################################################################
#This code was written by Ben Wilson with the supervison of Paul Ledger
#at the ZCCE, Swansea University
########################################################################

#Importing
import os
import numpy as np


def FeatureCreation(Tensors,Eigenvalues):
    Num_Freq = np.shape(Tensors)[0]
    Principal = np.zeros([Num_Freq,6])
    Deviatoric = np.zeros([Num_Freq,4])
    Z = np.zeros([Num_Freq])
    
    
    #Calculate the principal invarients
    #Invarient 1
    Principal[:,0] = (Tensors[:,0]+Tensors[:,4]+Tensors[:,8]).real
    Principal[:,1] = (Tensors[:,0]+Tensors[:,4]+Tensors[:,8]).imag
    #Invarient 2
    for i in range(Num_Freq):
        Matrix = Tensors[i,:].real.reshape((3,3))
        Matrix2 = Matrix@Matrix
        Principal[i,2] = (1/2)*(((Matrix[0,0]+Matrix[1,1]+Matrix[2,2])**2)-(Matrix2[0,0]+Matrix2[1,1]+Matrix2[2,2]))
    for i in range(Num_Freq):
        Matrix = Tensors[i,:].imag.reshape((3,3))
        Matrix2 = Matrix@Matrix
        Principal[i,3] = (1/2)*(((Matrix[0,0]+Matrix[1,1]+Matrix[2,2])**2)-(Matrix2[0,0]+Matrix2[1,1]+Matrix2[2,2]))
    #Invarient 3
    for i in range(Num_Freq):
        Matrix = Tensors[i,:].real.reshape((3,3))
        Principal[i,4] = np.linalg.det(Matrix)
    for i in range(Num_Freq):
        Matrix = Tensors[i,:].imag.reshape((3,3))
        Principal[i,5] = np.linalg.det(Matrix)
    
    
    
    #Calculate the deviatoric invarients
    #Invarient 2
    Deviatoric[:,0] = ((1/3)*Principal[:,0]**2)-Principal[:,2]
    Deviatoric[:,1] = ((1/3)*Principal[:,1]**2)-Principal[:,3]
    #Invarient 3
    Deviatoric[:,2] = ((2/27)*Principal[:,0]**3)-((1/3)*Principal[:,0]*Principal[:,2])+Principal[:,4]
    Deviatoric[:,3] = ((2/27)*Principal[:,1]**3)-((1/3)*Principal[:,1]*Principal[:,3])+Principal[:,5]
    
    
    
    #Calculate the comutator
    for i in range(Num_Freq):
        Real = Tensors[i,:].real.reshape((3,3))
        Imag = Tensors[i,:].imag.reshape((3,3))
        TempZ = Real@Imag-Imag@Real
        TempZ = TempZ@TempZ
        Z[i] = ((TempZ[0,1]**2)+(TempZ[0,2]**2)+(TempZ[1,2]**2))**(1/2)
    
    
    
    return Principal, Deviatoric, Z