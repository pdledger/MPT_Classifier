########################################################################
#This code was written by Ben Wilson with the supervison of Paul Ledger
#at the ZCCE, Swansea University
#Updated to compute invairants from eigenvalues since the noise is now added
#to eigenvalues. August 2023.
########################################################################

#Importing
import os
import numpy as np


def FeatureCreation(Tensors,Eigenvalues):
    Num_Freq = np.shape(Tensors)[0]
    Principal = np.zeros([Num_Freq,6])
    Deviatoric = np.zeros([Num_Freq,4])
    Z = np.zeros([Num_Freq])

    # Calculate the principal invarients
    #Invarient 1  lambda_1 + lambda_2 +lambda_3
    Principal[:,0] = (Eigenvalues[:,0]+Eigenvalues[:,1]+Eigenvalues[:,2]).real
    Principal[:,1] = (Eigenvalues[:,0]+Eigenvalues[:,1]+Eigenvalues[:,2]).imag

    #Invarient 2
    # lambda_1 * lambda_2+ lambda_1 *lambda_3 + lambda_2*lambda_3
    #Invariant 3
    # lambda_1 * lambda_2 * lambda_3
    for i in range(Num_Freq):
        eig1_real=Eigenvalues[i,0].real
        eig2_real=Eigenvalues[i,1].real
        eig3_real=Eigenvalues[i,2].real
        Principal[i,2] =eig1_real*eig2_real+ eig1_real*eig3_real+eig2_real*eig3_real
        Principal[i,4] =eig1_real*eig2_real*eig3_real
        
        
    for i in range(Num_Freq):
        eig1_imag=Eigenvalues[i,0].imag
        eig2_imag=Eigenvalues[i,1].imag
        eig3_imag=Eigenvalues[i,2].imag
        Principal[i,3] =eig1_imag*eig2_imag+ eig1_imag*eig3_imag+eig2_imag*eig3_imag
        Principal[i,5] =eig1_imag*eig2_imag*eig3_imag
       
    #Calculate the deviatoric invarients - not changed.
    #Invarient 2
    Deviatoric[:,0] = ((1/3)*Principal[:,0]**2)-Principal[:,2]
    Deviatoric[:,1] = ((1/3)*Principal[:,1]**2)-Principal[:,3]
    #Invarient 3
    Deviatoric[:,2] = ((2/27)*Principal[:,0]**3)-((1/3)*Principal[:,0]*Principal[:,2])+Principal[:,4]
    Deviatoric[:,3] = ((2/27)*Principal[:,1]**3)-((1/3)*Principal[:,1]*Principal[:,3])+Principal[:,5]

    # No update to commutator as needs noise adding to tensor coefficents.




##    Old method using tensor coefficents to compute invairants
##    
##    #Calculate the principal invarients
##    #Invarient 1
##    Principal[:,0] = (Tensors[:,0]+Tensors[:,4]+Tensors[:,8]).real
##    Principal[:,1] = (Tensors[:,0]+Tensors[:,4]+Tensors[:,8]).imag
##    #Invarient 2
##    for i in range(Num_Freq):
##        Matrix = Tensors[i,:].real.reshape((3,3))
##        Matrix2 = Matrix@Matrix
##        Principal[i,2] = (1/2)*(((Matrix[0,0]+Matrix[1,1]+Matrix[2,2])**2)-(Matrix2[0,0]+Matrix2[1,1]+Matrix2[2,2]))
##    for i in range(Num_Freq):
##        Matrix = Tensors[i,:].imag.reshape((3,3))
##        Matrix2 = Matrix@Matrix
##        Principal[i,3] = (1/2)*(((Matrix[0,0]+Matrix[1,1]+Matrix[2,2])**2)-(Matrix2[0,0]+Matrix2[1,1]+Matrix2[2,2]))
##    #Invarient 3
##    for i in range(Num_Freq):
##        Matrix = Tensors[i,:].real.reshape((3,3))
##        Principal[i,4] = np.linalg.det(Matrix)
##    for i in range(Num_Freq):
##        Matrix = Tensors[i,:].imag.reshape((3,3))
##        Principal[i,5] = np.linalg.det(Matrix)
##    
##    
##    
##    #Calculate the deviatoric invarients
##    #Invarient 2
##    Deviatoric[:,0] = ((1/3)*Principal[:,0]**2)-Principal[:,2]
##    Deviatoric[:,1] = ((1/3)*Principal[:,1]**2)-Principal[:,3]
##    #Invarient 3
##    Deviatoric[:,2] = ((2/27)*Principal[:,0]**3)-((1/3)*Principal[:,0]*Principal[:,2])+Principal[:,4]
##    Deviatoric[:,3] = ((2/27)*Principal[:,1]**3)-((1/3)*Principal[:,1]*Principal[:,3])+Principal[:,5]
##    
##    
##    
##    #Calculate the comutator
##    for i in range(Num_Freq):
##        Real = Tensors[i,:].real.reshape((3,3))
##        Imag = Tensors[i,:].imag.reshape((3,3))
##        TempZ = Real@Imag-Imag@Real
##        TempZ = TempZ@TempZ
##        Z[i] = ((TempZ[0,1]**2)+(TempZ[0,2]**2)+(TempZ[1,2]**2))**(1/2)
    
    
    
    return Principal, Deviatoric, Z
