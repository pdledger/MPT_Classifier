import numpy as np
from numpy import linalg as LA

from PreProcessors import *
def AGWN(Vec,SNR_db):

    # Vec = complex MPT coefficient as a function of omega
    # SNR_db = signal to noise ratio for power in dB    

    #Calculate the signal power
    sig_watts = (Vec * np.conjugate(Vec)).real
    sig_db = 10 * np.log10(sig_watts)

    #Calculate the noise
    noise_db = sig_db - SNR_db * np.ones(len(Vec))
    noise_power = 10 ** (noise_db / 10)

    #Calculate the noise to be added to the signal
    noise_volts = ((noise_power)/2)**(1/2) * (np.random.normal(0, 1, len(Vec)) +1j * np.random.normal(0, 1, len(Vec)))
    
    return Vec+noise_volts

def AGWNr(Vec,SNR_db):

    # Vec = real signal as a function of omega
    # SNR_db = signal to noise ratio for power in dB    

    #Calculate the signal power
    sig_watts = (Vec * np.conjugate(Vec)).real # dot product
    sig_db = 10 * np.log10(sig_watts)

    #Calculate the noise
    noise_db = sig_db - SNR_db * np.ones(len(Vec))
    noise_power = 10 ** (noise_db / 10)

    #Calculate the noise to be added to the signal
    noise_volts = (noise_power)**(1/2) * (np.random.normal(0, 1, len(Vec))) # return real valued noise
    
    return Vec+noise_volts


def Add_Noise(X_train,Training_noise,Tensors,Features,Frequencies,Feature_Dic):
    if type(Training_noise)!=str:
        for i in range(len(X_train)):

            # We have found that preallocating Tensor and then assigning Tensors to explicit indices reduces the
            # final error bars in the final results. This is thought to be due to a pointer error overwriting
            # Tensors with noisy data.
            Tensor = np.zeros((len(Frequencies), 9), dtype=complex)
            # Changed to add noise to eigenvalues. In practice we want to add noise to voltages, but since only
            # measured eigenvalues are available, it makes sense to add noise to simulated eigenvalues
            
            Tensor[:,:] = Tensors[int(X_train[i,0]),:,:]
            Eigenvalues = np.zeros([len(Frequencies),3],dtype=complex)
            NoisyEigenvalues = np.zeros([len(Frequencies),3],dtype=complex)
            
            for j in range(len(Frequencies)):
                Eigenvalues[j,:] = np.sort(LA.eig(Tensor[j,:].reshape(3,3).real)[0])
                Eigenvalues[j,:] += np.sort(1j*LA.eig(Tensor[j,:].reshape(3,3).imag)[0])
            NoisyEigenvalues[:,0] = AGWNr(np.real(Eigenvalues[:,0]),Training_noise)+1j*AGWNr(np.imag(Eigenvalues[:,0]),Training_noise)
            NoisyEigenvalues[:,1] = AGWNr(np.real(Eigenvalues[:,1]),Training_noise)+1j*AGWNr(np.imag(Eigenvalues[:,1]),Training_noise)
            NoisyEigenvalues[:,2] = AGWNr(np.real(Eigenvalues[:,2]),Training_noise)+1j*AGWNr(np.imag(Eigenvalues[:,2]),Training_noise)
            
            #Tensor[:,0] = AGWN(Tensor[:,0],Training_noise)
            #Tensor[:,4] = AGWN(Tensor[:,4],Training_noise)
            #Tensor[:,8] = AGWN(Tensor[:,8],Training_noise)
            #Tensor[:,1] = AGWN(Tensor[:,1],Training_noise)
            #Tensor[:,3] = Tensor[:,1]
            #Tensor[:,2] = AGWN(Tensor[:,2],Training_noise)
            #Tensor[:,6] = Tensor[:,2]
            #Tensor[:,5] = AGWN(Tensor[:,5],Training_noise)
            #Tensor[:,7] = Tensor[:,5]
            #Eigenvalues = np.zeros([len(Frequencies),3],dtype=complex)
            #for j in range(len(Frequencies)):
            #    Eigenvalues[j,:] = np.sort(LA.eig(Tensor[j,:].reshape(3,3).real)[0])
            #    Eigenvalues[j,:] += np.sort(1j*LA.eig(Tensor[j,:].reshape(3,3).imag)[0])
            # Previous implementation of FeatureCreation computed invairants computed from tensor coefficents
            # this has been updated to compute invariants from eigenvalues as noise has been added to eigenvalues.
            #Principal, Deviatoric, Z = FeatureCreation(Tensor,Eigenvalues)
            Principal, Deviatoric, Z = FeatureCreation(Tensor,NoisyEigenvalues)
            
            # Temp_data = np.zeros([17*len(Frequencies)])
            #NewData = np.concatenate((Eigenvalues[:, 0].real, Eigenvalues[:, 0].imag))
            #NewData = np.concatenate((NewData, Eigenvalues[:, 1].real, Eigenvalues[:, 1].imag))
            #NewData = np.concatenate((NewData, Eigenvalues[:, 2].real, Eigenvalues[:, 2].imag))
            NewData = np.concatenate((NoisyEigenvalues[:, 0].real, NoisyEigenvalues[:, 0].imag))
            NewData = np.concatenate((NewData, NoisyEigenvalues[:, 1].real, NoisyEigenvalues[:, 1].imag))
            NewData = np.concatenate((NewData, NoisyEigenvalues[:, 2].real, NoisyEigenvalues[:, 2].imag))

            NewData = np.concatenate((NewData, Principal[:, 0], Principal[:, 1], Principal[:, 2]))
            NewData = np.concatenate((NewData, Principal[:, 3], Principal[:, 4], Principal[:, 5]))
            NewData = np.concatenate((NewData, Deviatoric[:, 0], Deviatoric[:, 1], Deviatoric[:, 2], Deviatoric[:, 3]))
            NewData = np.concatenate((NewData, Z))

            for j,Feature in enumerate(Features):
                X_train[i,(len(Frequencies)*2*j)+1:(len(Frequencies)*2*(j+1))+1] = NewData[len(Frequencies)*2*Feature_Dic[Feature]:len(Frequencies)*2*(Feature_Dic[Feature]+1)]
    else:
        percentage, train_noise = Training_noise.split('%')
        percentage, train_noise = float(percentage)/100,float(train_noise)
        for i in range(len(X_train)):
            if np.random.binomial(1,percentage,1)==1:
                # We have found that preallocating Tensor and then assigning Tensors to explicit indices reduces the
                # final error bars in the final results. This is thought to be due to a pointer error overwriting
                # Tensors with noisy data.
                Tensor = np.zeros((len(Frequencies), 9), dtype=complex)
                Tensor[:, :] = Tensors[int(X_train[i, 0]), :, :]
               # Changed to add noise to eigenvalues. In practice we want to add noise to voltages, but since only
               # measured eigenvalues are available, it makes sense to add noise to simulated eigenvalues
                NoisyEigenvalues = np.zeros([len(Frequencies),3],dtype=complex)            
                for j in range(len(Frequencies)):
                    Eigenvalues[j,:] = np.sort(LA.eig(Tensor[j,:].reshape(3,3).real)[0])
                    Eigenvalues[j,:] += np.sort(1j*LA.eig(Tensor[j,:].reshape(3,3).imag)[0])
                NoisyEigenvalues[:,0] = AGWNr(np.real(Eigenvalues[:,0]),Training_noise)+1j*AGWNr(np.imag(Eigenvalues[:,0]),Training_noise)
                NoisyEigenvalues[:,1] = AGWNr(np.real(Eigenvalues[:,1]),Training_noise)+1j*AGWNr(np.imag(Eigenvalues[:,1]),Training_noise)
                NoisyEigenvalues[:,2] = AGWNr(np.real(Eigenvalues[:,2]),Training_noise)+1j*AGWNr(np.imag(Eigenvalues[:,2]),Training_noise)

                
                # Tensor = Tensors[int(X_train[i,0]),:,:]
                #Tensor[:,0] = AGWN(Tensor[:,0],train_noise)
                #Tensor[:,4] = AGWN(Tensor[:,4],train_noise)
                #Tensor[:,8] = AGWN(Tensor[:,8],train_noise)
                #Tensor[:,1] = AGWN(Tensor[:,1],train_noise)
                #Tensor[:,3] = Tensor[:,1]
                #Tensor[:,2] = AGWN(Tensor[:,2],train_noise)
                #Tensor[:,6] = Tensor[:,2]
                #Tensor[:,5] = AGWN(Tensor[:,5],train_noise)
                #Tensor[:,7] = Tensor[:,5]
                #Eigenvalues = np.zeros([len(Frequencies),3],dtype=complex)
                #for j in range(len(Frequencies)):
                #    Eigenvalues[j,:] = np.sort(LA.eig(Tensor[j,:].reshape(3,3).real)[0])
                #    Eigenvalues[j,:] += np.sort(1j*LA.eig(Tensor[j,:].reshape(3,3).imag)[0])
                #Principal, Deviatoric, Z = FeatureCreation(Tensor,Eigenvalues)
                # Temp_data = np.zeros([17*len(Frequencies)])
                # Previous implementation of FeatureCreation computed invairants computed from tensor coefficents
                # this has been updated to compute invariants from eigenvalues as noise has been added to eigenvalues.
                #Principal, Deviatoric, Z = FeatureCreation(Tensor,Eigenvalues)
                Principal, Deviatoric, Z = FeatureCreation(Tensor,NoisyEigenvalues)
 
                #NewData = np.concatenate((Eigenvalues[:, 0].real, Eigenvalues[:, 0].imag))
                #NewData = np.concatenate((NewData, Eigenvalues[:, 1].real, Eigenvalues[:, 1].imag))
                #NewData = np.concatenate((NewData, Eigenvalues[:, 2].real, Eigenvalues[:, 2].imag))
                NewData = np.concatenate((NoisyEigenvalues[:, 0].real, NoisyEigenvalues[:, 0].imag))
                NewData = np.concatenate((NewData, NoisyEigenvalues[:, 1].real, NoisyEigenvalues[:, 1].imag))
                NewData = np.concatenate((NewData, NoisyEigenvalues[:, 2].real, NoisyEigenvalues[:, 2].imag))
                
                NewData = np.concatenate((NewData, Principal[:, 0], Principal[:, 1], Principal[:, 2]))
                NewData = np.concatenate((NewData, Principal[:, 3], Principal[:, 4], Principal[:, 5]))
                NewData = np.concatenate(
                    (NewData, Deviatoric[:, 0], Deviatoric[:, 1], Deviatoric[:, 2], Deviatoric[:, 3]))
                NewData = np.concatenate((NewData, Z))

                for j,Feature in enumerate(Features):
                    X_train[i,(len(Frequencies)*2*j)+1:(len(Frequencies)*2*(j+1))+1] = NewData[len(Frequencies)*2*Feature_Dic[Feature]:len(Frequencies)*2*(Feature_Dic[Feature]+1)]
    return X_train[:,1:]
