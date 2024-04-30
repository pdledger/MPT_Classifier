import numpy as np
from numpy import linalg as LA

def ApplySVD (X_test_norm,X_train_norm,Testing_noise,DataSet_Name,Model,Savename ):

    # Generate matrix of data
    # nb X_test_norm has diemnsion N  x F
    # N = number of data samples
    # F = number of features
    N, F = np.shape(X_train_norm)
    print(N,F)

    # Create transpose and apply SVD

    M_test = np.transpose(X_test_norm)
    M_train = np.transpose(X_train_norm)

    MuTruncated, Ms, Mvh = np.linalg.svd(M_train, full_matrices=False)
    # Print an update on progress
    print(' SVD complete      ')

    # scale the value of the modes
    Msnorm = Ms / Ms[0]

    # Decide where to truncate
    cutoff = np.min((N,F))
    Tol = 1e-8
    for i in range(cutoff):
        if Msnorm[i] < Tol:
            cutoff = i
            break

    import matplotlib.pyplot as plt
    plt.semilogy(Msnorm)
    plt.xlabel("$n$")
    plt.ylabel("$(\Sigma)_{nn}/(\Sigma)_{11}$")

    if type(Testing_noise) == bool:
        #plt.savefig('Results/'+DataSet_Name+'/Noiseless/'+Model+'/'+Savename+'/posthistdistribution'+str(count)+'.pdf')
        plt.savefig('Results/'+DataSet_Name+'/Noiseless/'+Model+'/'+Savename+'/modelreductionsing.pdf')
    else:
        #plt.savefig('Results/'+DataSet_Name+'/Noise_'+str(Testing_noise)+'/'+Model+'/'+Savename+'/posthistdistribution'+str(count)+'.pdf')
        plt.savefig('Results/'+DataSet_Name+'/Noise_'+str(Testing_noise)+'/'+Model+'/'+Savename+'/modelreductionsing.pdf')
    plt.close()

    # Truncate the SVD matrices
    MuTruncated = MuTruncated[:, :cutoff]
    print("Reduced the number of features from",F," to ",cutoff)
    if cutoff < F : 

        # Create MuTruncated^T * M_train
        MuTtM_train = np.transpose(MuTruncated) @ M_train
        # Take the tranpose so that in the form
        # N x cutoff with cutoff being the new number of features
        X_train_norm_truncated = np.transpose(MuTtM_train)
        print(np.shape(X_train_norm_truncated))

        #Do the preprocessing
        X_Means = np.mean(X_train_norm_truncated,axis=0)
        X_SD = np.std(X_train_norm_truncated,axis=0)
        X_train_norm_truncated = (X_train_norm_truncated-X_Means)/X_SD

        # Create MuTruncated^T * M_test
        MuTtM_test = np.transpose(MuTruncated) @ M_test
        # Take the tranpose so that in the form
        # Noutput x cutoff with cutoff being the new number of features
        X_test_norm_truncated = np.transpose(MuTtM_test)
        X_test_norm_truncated = (X_test_norm_truncated-X_Means)/X_SD
    else:
        X_Means = np.mean(X_train_norm,axis=0)
        X_SD = np.std(X_train_norm,axis=0)
        X_test_norm_truncated = (X_test_norm-X_Means)/X_SD
        X_train_norm_truncated = (X_train_norm-X_Means)/X_SD

        

    return X_train_norm_truncated, X_test_norm_truncated
