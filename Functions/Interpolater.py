########################################################################
#This code was written by Ben Wilson with the supervison of Paul Ledger
#at the ZCCE, Swansea University
#Powered by NETGEN/NGSolve
########################################################################



#Import
import numpy as np
from scipy.interpolate import interp1d

#For Bishop's stuff
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

#Function to interpolate spectral data
def Interp(Orig_Freq,Eigenvalues,Tensors,Frequencies,cubic):
    #Load the data
    #Orig_Freq = np.genfromtxt(Filepath+'/Data/Frequencies.csv',delimiter=',')
    #Eigenvalues = np.genfromtxt(Filepath+'/Data/Eigenvalues.csv',delimiter=',',dtype=complex)
    #Tensors = np.genfromtxt(Filepath+'/Data/Tensors.csv',delimiter=',',dtype=complex)
    
    #Split the output frequencies
    Low_freq = [x for x in Frequencies if x<min(Orig_Freq)]
    Mid_freq = [x for x in Frequencies if x>=min(Orig_Freq) and x<=max(Orig_Freq)]
    High_freq = [x for x in Frequencies if x>max(Orig_Freq)]
    
    #Create a place to save data
    Output_Tensors = np.zeros([len(Frequencies),9],dtype=complex)
    Output_Eigenvalues = np.zeros([len(Frequencies),3],dtype=complex)
    Mid_Ten = np.zeros([len(Mid_freq),9],dtype=complex)
    Mid_Eig = np.zeros([len(Mid_freq),3],dtype=complex)
    #Interpolate
    Low_Ten = [Tensors[0] for x in Low_freq]
    High_Ten = [Tensors[-1] for x in High_freq]
    Low_Eig = [Eigenvalues[0] for x in Low_freq]
    High_Eig = [Eigenvalues[-1] for x in High_freq]
    for i in range(9):
        if cubic == True:
            Ten_fun = interp1d(Orig_Freq,Tensors[:,i],kind='cubic')
        else:
            Ten_fun = interp1d(Orig_Freq,Tensors[:,i])
        if i<3:
            if cubic == True:
                Eig_fun = interp1d(Orig_Freq,Eigenvalues[:,i],kind='cubic')
            else:
                Eig_fun = interp1d(Orig_Freq,Eigenvalues[:,i])
    
        Mid_Ten[:,i] = Ten_fun(Mid_freq)
        if i<3:
            Mid_Eig[:,i] = Eig_fun(Mid_freq)
    
    if Low_Ten == [] and High_Ten == []:
        Output_Tensors = Mid_Ten
        Output_Eigenvalues = Mid_Eig
    elif Low_Ten == []:
        Output_Tensors = np.concatenate((Mid_Ten,High_Ten), axis=0)
        Output_Eigenvalues = np.concatenate((Mid_Eig,High_Eig), axis=0)
    elif High_Ten == []:
        Output_Tensors = np.concatenate((Low_Ten,Mid_Ten), axis=0)
        Output_Eigenvalues = np.concatenate((Low_Eig,Mid_Eig), axis=0)
    else:
        Output_Tensors = np.concatenate((np.concatenate((Low_Ten,Mid_Ten), axis=0),High_Ten), axis=0)
        Output_Eigenvalues = np.concatenate((np.concatenate((Low_Eig,Mid_Eig), axis=0),High_Eig), axis=0)
    
    return Output_Tensors,Output_Eigenvalues

#Function to interpolate spectral data
def LogInterp(Orig_Freq,Eigenvalues,Tensors,Frequencies,cubic):
    #Load the data
    #Orig_Freq = np.genfromtxt(Filepath+'/Data/Frequencies.csv',delimiter=',')
    #Eigenvalues = np.genfromtxt(Filepath+'/Data/Eigenvalues.csv',delimiter=',',dtype=complex)
    #Tensors = np.genfromtxt(Filepath+'/Data/Tensors.csv',delimiter=',',dtype=complex)
    Orig_Freq = np.log(Orig_Freq)
    Frequencies = np.log(Frequencies)
    
    #Split the output frequencies
    Low_freq = [x for x in Frequencies if x<min(Orig_Freq)]
    Mid_freq = [x for x in Frequencies if x>=min(Orig_Freq) and x<=max(Orig_Freq)]
    High_freq = [x for x in Frequencies if x>max(Orig_Freq)]
    
    #Create a place to save data
    Output_Tensors = np.zeros([len(Frequencies),9],dtype=complex)
    Output_Eigenvalues = np.zeros([len(Frequencies),3],dtype=complex)
    Mid_Ten = np.zeros([len(Mid_freq),9],dtype=complex)
    Mid_Eig = np.zeros([len(Mid_freq),3],dtype=complex)
    #Interpolate
    Low_Ten = [Tensors[0] for x in Low_freq]
    High_Ten = [Tensors[-1] for x in High_freq]
    Low_Eig = [Eigenvalues[0] for x in Low_freq]
    High_Eig = [Eigenvalues[-1] for x in High_freq]
    for i in range(9):
        if cubic == True:
            Ten_fun = interp1d(Orig_Freq,Tensors[:,i],kind='cubic')
        else:
            Ten_fun = interp1d(Orig_Freq,Tensors[:,i])
        if i<3:
            if cubic == True:
                Eig_fun = interp1d(Orig_Freq,Eigenvalues[:,i],kind='cubic')
            else:
                Eig_fun = interp1d(Orig_Freq,Eigenvalues[:,i])
    
        Mid_Ten[:,i] = Ten_fun(Mid_freq)
        if i<3:
            Mid_Eig[:,i] = Eig_fun(Mid_freq)
    
    if Low_Ten == [] and High_Ten == []:
        Output_Tensors = Mid_Ten
        Output_Eigenvalues = Mid_Eig
    elif Low_Ten == []:
        Output_Tensors = np.concatenate((Mid_Ten,High_Ten), axis=0)
        Output_Eigenvalues = np.concatenate((Mid_Eig,High_Eig), axis=0)
    elif High_Ten == []:
        Output_Tensors = np.concatenate((Low_Ten,Mid_Ten), axis=0)
        Output_Eigenvalues = np.concatenate((Low_Eig,Mid_Eig), axis=0)
    else:
        Output_Tensors = np.concatenate((np.concatenate((Low_Ten,Mid_Ten), axis=0),High_Ten), axis=0)
        Output_Eigenvalues = np.concatenate((np.concatenate((Low_Eig,Mid_Eig), axis=0),High_Eig), axis=0)
    
    return Output_Tensors,Output_Eigenvalues



def InterpBish(Filepath,Frequencies,degree,alpha):
    #Load the data
    Orig_Freq = np.genfromtxt(Filepath+'/Data/Frequencies.csv',delimiter=',')
    Eigenvalues = np.genfromtxt(Filepath+'/Data/Eigenvalues.csv',delimiter=',',dtype=complex)
    Tensors = np.genfromtxt(Filepath+'/Data/Tensors.csv',delimiter=',',dtype=complex)
    
    #Create a place to save data
    Output_Tensors = np.zeros([len(Frequencies),9],dtype=complex)
    Output_Eigenvalues = np.zeros([len(Frequencies),3],dtype=complex)
    
    #Make the model
    est = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=alpha))
    for i in range(3):
        est.fit(Orig_Freq.reshape(-1,1), Eigenvalues[:,i].real.reshape(-1,1))
        Output_Eigenvalues[:,i] = est.predict(Frequencies.reshape(-1,1)).flatten()
        est.fit(Orig_Freq.reshape(-1,1), Eigenvalues[:,i].imag.reshape(-1,1))
        Output_Eigenvalues[:,i] += (est.predict(Frequencies.reshape(-1,1))*1j).flatten()
    
    for i in range(9):
        est.fit(Orig_Freq.reshape(-1,1), Tensors[:,i].real.reshape(-1,1))
        Output_Tensors[:,i] = est.predict(Frequencies.reshape(-1,1)).flatten()
        est.fit(Orig_Freq.reshape(-1,1), Tensors[:,i].imag.reshape(-1,1))
        Output_Tensors[:,i] += (est.predict(Frequencies.reshape(-1,1))*1j).flatten()
        
    
    
    
    
    return Output_Tensors,Output_Eigenvalues

def InterpLogBish(Filepath,Frequencies,degree,alpha):
    #Load the data
    Orig_Freq = np.genfromtxt(Filepath+'/Data/Frequencies.csv',delimiter=',')
    Eigenvalues = np.genfromtxt(Filepath+'/Data/Eigenvalues.csv',delimiter=',',dtype=complex)
    Tensors = np.genfromtxt(Filepath+'/Data/Tensors.csv',delimiter=',',dtype=complex)
    
    #Create a place to save data
    Output_Tensors = np.zeros([len(Frequencies),9],dtype=complex)
    Output_Eigenvalues = np.zeros([len(Frequencies),3],dtype=complex)
    
    #Make the model
    est = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=alpha))
    for i in range(3):
        #print(np.log(Orig_Freq))
        est.fit(np.log(Orig_Freq).reshape(-1,1), Eigenvalues[:,i].real.reshape(-1,1))
        Output_Eigenvalues[:,i] = est.predict(np.log(Frequencies).reshape(-1,1)).flatten()
        est.fit(np.log(Orig_Freq).reshape(-1,1), Eigenvalues[:,i].imag.reshape(-1,1))
        Output_Eigenvalues[:,i] += (est.predict(np.log(Frequencies).reshape(-1,1))*1j).flatten()
    
    for i in range(9):
        est.fit(Orig_Freq.reshape(-1,1), Tensors[:,i].real.reshape(-1,1))
        Output_Tensors[:,i] = est.predict(np.log(Frequencies).reshape(-1,1)).flatten()
        est.fit(Orig_Freq.reshape(-1,1), Tensors[:,i].imag.reshape(-1,1))
        Output_Tensors[:,i] += (est.predict(np.log(Frequencies).reshape(-1,1))*1j).flatten()
        
    
    
    
    
    return Output_Tensors,Output_Eigenvalues
    
    
    
    




























