import numpy as np
import scipy
import jax

def EigenValueDecomposition(N0,TensorArray,Frequencies):

    N = len(Frequencies)
    MultRstore=np.zeros(N)
    MultIstore=np.zeros(N)
    MultRtildestore=np.zeros(N)
    MultN0store=np.zeros(N) # repeated storage of N0 multplicities, eigenvectors, eigenvalues etc

    URstore=np.zeros((N,3),dtype=np.longdouble)
    QRstore=np.zeros((N,3,3),dtype=np.longdouble)
    UIstore=np.zeros((N,3),dtype=np.longdouble)
    QIstore=np.zeros((N,3,3),dtype=np.longdouble)
    UN0store=np.zeros((N,3),dtype=np.longdouble)
    QN0store=np.zeros((N,3,3),dtype=np.longdouble)
    URtildestore=np.zeros((N,3),dtype=np.longdouble)
    QRtildestore=np.zeros((N,3,3),dtype=np.longdouble)
    tol=1e-4
    for n in range(N):
        Mlist = TensorArray[n,:]
        Mten = np.array([[Mlist[0], Mlist[1], Mlist[2]],[Mlist[3], Mlist[4], Mlist[5]],[Mlist[6], Mlist[7], Mlist[8]]],dtype=np.clongdouble)
        Rtilde = np.real(Mten)#+np.diag(np.random.rand(3))*tol
        I = np.imag(Mten)#+np.diag(np.random.rand(3))*tol
        R = (np.real(Mten)-N0)

        # Computation of eigenvalues, eigenvector
        # Note that the we need to apply to normal matrices and so consider R^TR = R R = R^2
        # Note that the sqrt( eigenvalues of R^2) are the abs(lambda(R))
        # The eigenvectors of R^2 are the same as the eigenvector of R
        #uR,VR = np.linalg.eig((R@R).astype(dtype=float))
        #uRtilde,VRtilde = np.linalg.eig((Rtilde@Rtilde).astype(dtype=float))
        #uI,VI = np.linalg.eig((I@I).astype(dtype=float))
        #uN0,VN0 = np.linalg.eig((N0@N0).astype(dtype=float))
        #uR=np.sqrt(uR)
        #uRtilde=np.sqrt(uRtilde)
        #uI=np.sqrt(uI)
        #uN0=np.sqrt(uN0)

        uR,VR = np.linalg.eig(R.astype(dtype=float))#jax.numpy.linalg.eig(R)
        uR=np.real(uR)
        VR=np.real(VR)
        #print(uR,VR)
        uRtilde,VRtilde = np.linalg.eig(Rtilde.astype(dtype=float))#jax.numpy.linalg.eig(Rtilde)
        uRtilde=np.real(uRtilde)
        VRtilde=np.real(VRtilde)
        uI,VI = np.linalg.eig(I.astype(dtype=float))#jax.numpy.linalg.eig(I)
        uI=np.real(uI)
        VI=np.real(VI)
        #print(uI)
        uN0,VN0 = np.linalg.eig(N0.astype(dtype=float))#jax.numpy.linalg.eig(N0)
        uN0=np.real(uN0)
        VN0=np.real(VN0)
        #uR=np.abs(uR)
        #uRtilde=np.abs(uRtilde)
        #uI=np.abs(uI)
        #uN0=np.abs(uN0)

        # Sort the eigenvalues from largest to smallest magnitude
        # ind = np.argsort(-np.abs(uR))
        # uR = uR[ind]
        # VR = VR[:,ind]
        # ind = np.argsort(-np.abs(uI))
        # uI = uI[ind]
        # VI = VI[:,ind]
        # ind = np.argsort(-np.abs(uRtilde))
        # uRtilde = uRtilde[ind]
        # VRtilde = VRtilde[:,ind]
        # ind = np.argsort(-np.abs(uN0))
        # uN0 = uN0[ind]
        # VN0 = VN0[:,ind]
        # Sort the eigenvalues from largest to smallest

#        ind = np.argsort(-(uR))
#        uR = uR[ind]
#        VR = VR[:,ind]

#        ind = np.argsort(-(uI))
#        uI = uI[ind]
#        VI = VI[:,ind]
#        ind = np.argsort(-(uRtilde))
#        uRtilde = uRtilde[ind]
#        VRtilde = VRtilde[:,ind]
#        ind = np.argsort(-uN0)
#        uN0 = uN0[ind]
#        VN0 = VN0[:,ind]


    # Possible multiplicities are 1, 2 or 3
    # If they are order in terms of increasing multiplicities we will either have
    #  case 1 : 1 , 1, 1 if they are distinct
    #  case 2 : or 1, 2
    #  case 3 : or 3
        MultR = CheckMult(uR,R.astype(dtype=float))
        MultI = CheckMult(uI,I.astype(dtype=float))
        MultRtilde = CheckMult(uRtilde,Rtilde.astype(dtype=float))
        MultN0 = CheckMult(uN0,N0.astype(dtype=float))


        #if n > 5 and n < 8:
        #    print(n)
        #    print("R",uR,VR,MultR)
        #    print("Rtilde",uRtilde,VRtilde,MultRtilde)
        #    print("I",uI,VI,MultI)


        MultRstore[n] = MultR
        MultIstore[n] = MultI
        MultRtildestore[n] = MultRtilde
        MultN0store[n] = MultN0



        # Store eigenvalues and eigenvectors
        for i in range(3):
            URstore[n,i]=uR[i]
            UIstore[n,i]=uI[i]
            URtildestore[n,i]=uRtilde[i]
            UN0store[n,i]=uN0[i]

            for j in range(3):
                QRstore[n,i,j]=VR[i,j]
                QIstore[n,i,j]=VI[i,j]
                QRtildestore[n,i,j]=VRtilde[i,j]
                QN0store[n,i,j]=VN0[i,j]

    return MultRstore, MultIstore, MultRtildestore, MultN0store, URstore, UIstore, URtildestore, UN0store, QRstore, QIstore, QRtildestore, QN0store

def CheckMult(u,Tensor):
    # Dynamically adjust tolerance
    Tol=1e-4*np.min(np.abs(u))
    mult=0
#    # Determine the multplicity of eigenvalue lambda_i as 3 - rank(R -lambda_i eye(3))
    for i in range(3):
        mult=np.max([mult,3-np.linalg.matrix_rank(Tensor-u[i]*np.eye(3),tol=Tol)])
#
    # Tol=5e-4
    # mult=1
    # for i in range(3):
    #     for j in range(i+1,3):
    #         if i != j:
    #             if np.abs(u[i]-u[j])/abs(u[i]) < Tol:
    #                 mult=mult+1
    #                 if mult > 3:
    #                     mult =3

    return mult
