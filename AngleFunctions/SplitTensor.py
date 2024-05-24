import numpy as np

def SplitTensor(TensorArray,Frequencies,N0):
    N=len(Frequencies)
    # Split in to R, I and Rtilde
    Rstore = np.zeros((N,3,3))
    Istore = np.zeros((N,3,3))
    Rtildestore = np.zeros((N,3,3))
    N0store = np.zeros((N,3,3))

    for n in range(N):
        Mlist = TensorArray[n,:]
        Mten = np.array([[Mlist[0], Mlist[1], Mlist[2]],[Mlist[3], Mlist[4], Mlist[5]],[Mlist[6], Mlist[7], Mlist[8]]])
        # Mten is generated from interpolation and scaling and so may not be symmetric ensure symmetry
        Mten = (Mten + np.transpose(Mten))/2.
        Rtilde = np.real(Mten)
        I = np.imag(Mten)
        R = np.real(Mten)-N0
        for i in range(3):
            for j in range(3):
                Rstore[n,i,j]=R[i,j]
                Istore[n,i,j]=I[i,j]
                Rtildestore[n,i,j]=Rtilde[i,j]
                N0store[n,i,j]=N0[i,j]


    return Rstore,Istore,Rtildestore, N0store
