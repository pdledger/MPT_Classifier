########################################################################
#This code was written by Ben Wilson with the supervison of Paul Ledger
#at the ZCCE, Swansea University
#Powered by NETGEN/NGSolve
########################################################################

#Importing
import os
import sys
import numpy as np


#Function to edit floats to a nice format
def FtoS(value):
    if value==0:
        newvalue = "0"
    elif value==1:
        newvalue = "1"
    elif value==-1:
        newvalue = "-1"
    else:
        for i in range(100):
            if abs(value)<=1:
                if round(abs(value/10**(-i)),2)>=1:
                    power=-i
                    break
            else:
                if round(abs(value/10**(i)),2)<1:
                    power=i-1
                    break
        newvalue=value/(10**power)
        newvalue=str(round(newvalue,2))
        if newvalue[-1]=="0":
            newvalue=newvalue[:-1]
        if newvalue[-1]=="0":
            newvalue=newvalue[:-1]
        if newvalue[-1]==".":
            newvalue=newvalue[:-1]
        newvalue += "e"+str(power)

    return newvalue


#Function to scale
def Scale(Object,AlphaValue,SigmaValue,AlphaSD,SigmaSD,Samples):
    
    AllFolders = os.listdir(Object)#List the folders in the object folder
    AllFolders = [Folder for Folder in AllFolders if '.DS' not in Folder]#Remove the folders we are not interested in
    
    Alphas = ['al_'+str(val) for val in AlphaValue] #Create list of strings to check through
    try:
        Sigmas = ['sig_'+FtoS(val) for val in SigmaValue] #Create list of strings to check through for floats
    except:
        Sigmas = ['sig_'+val for val in SigmaValue] #Create list of strings to check through for strings
    Folders = [Folder for Folder in AllFolders if any(Si in Folder for Si in Sigmas) and any(Al in Folder for Al in Alphas)] #Decide which folders to scale
    
    Alpha = True
    Sigma = True
    
    #Scale the resutls
    for Folder in Folders:
        #reread the folder name
        _,AlphaVal,_,Mu,_,SigmaVal = Folder.split('_')
    
        #Calculate the new values for Alpha and Sigma
        if Alpha == True:
            AlphaVal, AlSig = float(AlphaVal),float(AlphaVal)*AlphaSD*0.01
            AlScales = np.random.normal(1,0.01*AlphaSD,Samples)
            NewAl = AlScales*AlphaVal
        else:
            NewAl = np.ones(Samples)*float(AlphaVal)
        if Sigma == True:
            SigScales = np.random.normal(1,0.01*SigmaSD,Samples)
            if ',' in SigmaVal:
                DoubleSig = True
                SigmaVal = [float(x) for x in SigmaVal.split(',')]
                SigSig = [x*SigmaSD*0.01 for x in SigmaVal]
                NewSig = [[x*y for y in SigmaVal] for x in SigScales]
            else:
                DoubleSig = False
                SigmaVal, SigSig = float(SigmaVal),float(SigmaVal)*SigmaSD*0.01
                NewSig = [x*SigmaVal for x in SigScales]
        else:
            if ',' in SigmaVal:
                DoubleSig = True
                SigmaVal = [float(x) for x in SigmaVal.split(',')]
                NewSig = [[x*y for y in SigmaVal] for x in np.ones(Samples)]
            else:
                DoubleSig = False
                NewSig = np.ones(Samples)*float(SigmaVal)
    

        #For each of the Frequency sweeps
        Sweeps = os.listdir(Object+'/'+Folder)
        Sweeps = [Sweep for Sweep in Sweeps if '.DS' not in Sweep]
        for Sweep in Sweeps:
            #Load the data
            if 'om' in Sweep:
                _,Frequencies,a,b,c,d = Sweep.split('_')
                Frequencies = float(Frequencies)
                Tensors = np.genfromtxt(Object+'/'+Folder+'/'+Sweep+'/Data/MPT.csv',delimiter = ',',dtype = complex)
            else:
                Frequencies = np.genfromtxt(Object+'/'+Folder+'/'+Sweep+'/Data/Frequencies.csv',delimiter = ',')
                Tensors = np.genfromtxt(Object+'/'+Folder+'/'+Sweep+'/Data/Tensors.csv',delimiter = ',',dtype = complex)
            
            N0 = np.genfromtxt(Object+'/'+Folder+'/'+Sweep+'/Data/N0.csv',delimiter = ',')
            Eigenvalues = np.genfromtxt(Object+'/'+Folder+'/'+Sweep+'/Data/Eigenvalues.csv',delimiter = ',',dtype = complex)
            
            
            AllNewFrequencies = np.zeros([len(Frequencies),Samples])
            AllNewEigenvalues = np.zeros([len(Frequencies),3,Samples],dtype=complex)
            AllNewTensors = np.zeros([len(Frequencies),9,Samples],dtype=complex)
            AllNewAlphas = []
            AllNewSigmas = []
            

            #Create the new folders and add the data
            for i in range(Samples):
                #Create the new folder names
                if DoubleSig == True:
                    SigStr = ''.join([','+str(Sig) for Sig in NewSig[i]])
                    AllNewSigmas.append(SigStr[1:])
                else:
                    AllNewSigmas.append(str(NewSig[i]))
                AllNewAlphas.append(str(NewAl[i]))
                
                #Scale the results
                #For Sigma
                if Sigma == True:
                    NewFrequencies = Frequencies/SigScales[i]
                    if SigScales[i]<=0:
                        print(SigScales[i])
                else:
                    NewFrequencies = Frequencies
                #For Alpha
                if Alpha == True:
                    NewFrequencies = NewFrequencies/(AlScales[i]**2)
                    NewTensors = (AlScales[i]**3)*Tensors
                    NewEigenvalues = (AlScales[i]**3)*Eigenvalues
                    NewN0 = (AlScales[i]**3)*N0
                else:
                    NewFrequencies = NewFrequencies
                    NewTensors = Tensors
                    NewEigenvalues = Eigenvalues
                    NewN0 = N0
                
                
                #Save the data
                AllNewFrequencies[:,i] = NewFrequencies
                AllNewEigenvalues[:,:,i] = NewEigenvalues
                AllNewTensors[:,:,i] = NewTensors
    
    
    
    
    
    return Frequencies, Eigenvalues, Tensors, AllNewFrequencies, AllNewEigenvalues, AllNewTensors, AllNewAlphas, AllNewSigmas#, AlScales, SigScales





#Function to scale
def Box_Scale(Object,AlphaValue,SigmaValue,AlphaValue_Low,SigmaValue_Low,AlphaValue_High,SigmaValue_High,Samples):
    Samples = int(Samples/4)
    
    AllFolders = os.listdir(Object)#List the folders in the object folder
    AllFolders = [Folder for Folder in AllFolders if '.DS' not in Folder]#Remove the folders we are not interested in
    
    Alphas = ['al_'+str(val) for val in AlphaValue] #Create list of strings to check through
    try:
        Sigmas = ['sig_'+FtoS(val) for val in SigmaValue] #Create list of strings to check through for floats
    except:
        Sigmas = ['sig_'+val for val in SigmaValue] #Create list of strings to check through for strings
    Folders = [Folder for Folder in AllFolders if any(Si in Folder for Si in Sigmas) and any(Al in Folder for Al in Alphas)] #Decide which folders to scale
    
    Alpha = True
    Sigma = True
    
    #Scale the resutls
    for Folder in Folders:
        #reread the folder name
        _,AlphaVal,_,Mu,_,SigmaVal = Folder.split('_')
    
        #Calculate the new values for Alpha and Sigma
        if Alpha == True:
            AlphaVal = float(AlphaValue[0])
            AlScales = np.linspace(1+(0.01*AlphaValue_Low),1+(0.01*AlphaValue_High),Samples)
            NewAl = AlScales*AlphaVal
        else:
            NewAl = np.ones(Samples)*float(AlphaVal)
        if Sigma == True:
            if type(SigmaValue[0]) == str:
                SigmaVal = float(SigmaValue[0])
            else:
                SigmaVal = [float(i) for i in SigmaValue]
            SigmaVal = SigmaValue[0]
            SigScales = np.linspace(1+(0.01*SigmaValue_Low),1+(0.01*SigmaValue_High),Samples)
            if ',' in SigmaVal:
                DoubleSig = True
                SigmaVal = [float(x) for x in SigmaVal.split(',')]
                SigSig = [x*SigmaSD*0.01 for x in SigmaVal]
                NewSig = [[x*y for y in SigmaVal] for x in SigScales]
            else:
                DoubleSig = False
                SigmaVal = float(SigmaVal)
                NewSig = [x*SigmaVal for x in SigScales]
        else:
            if ',' in SigmaVal:
                DoubleSig = True
                SigmaVal = [float(x) for x in SigmaVal.split(',')]
                NewSig = [[x*y for y in SigmaVal] for x in np.ones(Samples)]
            else:
                DoubleSig = False
                NewSig = np.ones(Samples)*float(SigmaVal)
    

        #For each of the Frequency sweeps
        Sweeps = os.listdir(Object+'/'+Folder)
        Sweeps = [Sweep for Sweep in Sweeps if '.DS' not in Sweep]
        for Sweep in Sweeps:
            #Load the data
            if 'om' in Sweep:
                _,Frequencies,a,b,c,d = Sweep.split('_')
                Frequencies = float(Frequencies)
                Tensors = np.genfromtxt(Object+'/'+Folder+'/'+Sweep+'/Data/MPT.csv',delimiter = ',',dtype = complex)
            else:
                Frequencies = np.genfromtxt(Object+'/'+Folder+'/'+Sweep+'/Data/Frequencies.csv',delimiter = ',')
                Tensors = np.genfromtxt(Object+'/'+Folder+'/'+Sweep+'/Data/Tensors.csv',delimiter = ',',dtype = complex)
            
            N0 = np.genfromtxt(Object+'/'+Folder+'/'+Sweep+'/Data/N0.csv',delimiter = ',')
            Eigenvalues = np.genfromtxt(Object+'/'+Folder+'/'+Sweep+'/Data/Eigenvalues.csv',delimiter = ',',dtype = complex)
            
            
            AllNewFrequencies = np.zeros([len(Frequencies),Samples*4])
            AllNewEigenvalues = np.zeros([len(Frequencies),3,Samples*4],dtype=complex)
            AllNewTensors = np.zeros([len(Frequencies),9,Samples*4],dtype=complex)
            AllNewAlphas = []
            AllNewSigmas = []
            

            #Create the new folders and add the data
            #Alphas
            for i in range(Samples):
                #Create the new folder names
                if DoubleSig == True:
                    SigStr = ''.join([','+str(Sig) for Sig in NewSig[i]])
                    AllNewSigmas.append(SigStr[1:])
                else:
                    AllNewSigmas.append(str(np.amin(NewSig)))
                AllNewAlphas.append(str(NewAl[i]))
                
                #Scale the results
                #For Sigma
                if Sigma == True:
                    NewFrequencies = Frequencies/np.amin(SigScales)
                else:
                    NewFrequencies = Frequencies
                #For Alpha
                if Alpha == True:
                    NewFrequencies = NewFrequencies/(AlScales[i]**2)
                    NewTensors = (AlScales[i]**3)*Tensors
                    NewEigenvalues = (AlScales[i]**3)*Eigenvalues
                    NewN0 = (AlScales[i]**3)*N0
                else:
                    NewFrequencies = NewFrequencies
                    NewTensors = Tensors
                    NewEigenvalues = Eigenvalues
                    NewN0 = N0
                
                
                #Save the data
                AllNewFrequencies[:,i] = NewFrequencies
                AllNewEigenvalues[:,:,i] = NewEigenvalues
                AllNewTensors[:,:,i] = NewTensors
            
            #Create the new folders and add the data
            for i in range(Samples):
                #Create the new folder names
                if DoubleSig == True:
                    SigStr = ''.join([','+str(Sig) for Sig in NewSig[i]])
                    AllNewSigmas.append(SigStr[1:])
                else:
                    AllNewSigmas.append(str(np.amax(NewSig)))
                AllNewAlphas.append(str(NewAl[i]))
                
                #Scale the results
                #For Sigma
                if Sigma == True:
                    NewFrequencies = Frequencies/np.amax(SigScales)
                else:
                    NewFrequencies = Frequencies
                #For Alpha
                if Alpha == True:
                    NewFrequencies = NewFrequencies/(AlScales[i]**2)
                    NewTensors = (AlScales[i]**3)*Tensors
                    NewEigenvalues = (AlScales[i]**3)*Eigenvalues
                    NewN0 = (AlScales[i]**3)*N0
                else:
                    NewFrequencies = NewFrequencies
                    NewTensors = Tensors
                    NewEigenvalues = Eigenvalues
                    NewN0 = N0
                
                
                #Save the data
                AllNewFrequencies[:,Samples+i] = NewFrequencies
                AllNewEigenvalues[:,:,Samples+i] = NewEigenvalues
                AllNewTensors[:,:,Samples+i] = NewTensors
            
            #Sigmas
            for i in range(Samples):
                #Create the new folder names
                if DoubleSig == True:
                    SigStr = ''.join([','+str(Sig) for Sig in NewSig[i]])
                    AllNewSigmas.append(SigStr[1:])
                else:
                    AllNewSigmas.append(str(NewSig[i]))
                AllNewAlphas.append(str(np.amin(NewAl)))
                
                #Scale the results
                #For Sigma
                if Sigma == True:
                    NewFrequencies = Frequencies/SigScales[i]
                else:
                    NewFrequencies = Frequencies
                #For Alpha
                if Alpha == True:
                    NewFrequencies = NewFrequencies/(np.amin(AlScales)**2)
                    NewTensors = (np.amin(AlScales)**3)*Tensors
                    NewEigenvalues = (np.amin(AlScales)**3)*Eigenvalues
                    NewN0 = (np.amin(AlScales)**3)*N0
                else:
                    NewFrequencies = NewFrequencies
                    NewTensors = Tensors
                    NewEigenvalues = Eigenvalues
                    NewN0 = N0
                
                
                #Save the data
                AllNewFrequencies[:,2*Samples+i] = NewFrequencies
                AllNewEigenvalues[:,:,2*Samples+i] = NewEigenvalues
                AllNewTensors[:,:,2*Samples+i] = NewTensors
            
            #Create the new folders and add the data
            for i in range(Samples):
                #Create the new folder names
                if DoubleSig == True:
                    SigStr = ''.join([','+str(Sig) for Sig in NewSig[i]])
                    AllNewSigmas.append(SigStr[1:])
                else:
                    AllNewSigmas.append(str(NewSig[i]))
                AllNewAlphas.append(str(np.amax(NewAl)))
                
                #Scale the results
                #For Sigma
                if Sigma == True:
                    NewFrequencies = Frequencies/SigScales[i]
                else:
                    NewFrequencies = Frequencies
                #For Alpha
                if Alpha == True:
                    NewFrequencies = NewFrequencies/(np.amax(AlScales)**2)
                    NewTensors = (np.amax(AlScales)**3)*Tensors
                    NewEigenvalues = (np.amax(AlScales)**3)*Eigenvalues
                    NewN0 = (np.amax(AlScales)**3)*N0
                else:
                    NewFrequencies = NewFrequencies
                    NewTensors = Tensors
                    NewEigenvalues = Eigenvalues
                    NewN0 = N0
                
                
                #Save the data
                AllNewFrequencies[:,3*Samples+i] = NewFrequencies
                AllNewEigenvalues[:,:,3*Samples+i] = NewEigenvalues
                AllNewTensors[:,:,3*Samples+i] = NewTensors
    
    
    
    
    
    return Frequencies, Eigenvalues, Tensors, AllNewFrequencies, AllNewEigenvalues, AllNewTensors, AllNewAlphas, AllNewSigmas#, AlScales, SigScales
    