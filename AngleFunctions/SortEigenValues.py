import numpy as np
import time

from Rodrigues import *
def SortEigenValues(MultRstore, MultIstore, URstore, UIstore, QRstore, QIstore, Frequencies, sorteigenvalues, Rstore, Istore):
    N = len(Frequencies)
    # Prepare sorted values by making a copy (note multiplicties don't change)
    SortedMultRstore=np.copy(MultRstore)
    SortedMultIstore=np.copy(MultIstore)
    SortedURstore =np.zeros((N,3))
    SortedUIstore =np.zeros((N,3))
    SortedQRstore = np.zeros((N,3,3))
    SortedQIstore = np.zeros((N,3,3))
    SortedKstore = np.zeros((N,3,3))

    Perm = np.array([[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]])
    sign=np.array([[1,1,1], \
                        [-1,1,1], \
                        [-1,-1,1], \
                        [-1,-1,-1], \
                        [-1,1,-1], \
                        [1,-1,-1], \
                        [1,-1,1], \
                        [1,1,-1]])


    for n in range(N):
        QR = np.zeros((3,3),dtype=np.longdouble)
        QI = np.zeros((3,3),dtype=np.longdouble)
        R = np.zeros((3,3),dtype=np.longdouble)
        I = np.zeros((3,3),dtype=np.longdouble)

        uR =np.zeros(3,dtype=np.longdouble)
        uI = np.zeros(3,dtype=np.longdouble)
        for i in range(3):
            uR[i]=URstore[n,i]
            uI[i]=UIstore[n,i]

            for j in range(3):
                QR[i,j] = QRstore[n,i,j]
                QI[i,j] = QIstore[n,i,j]

                R[i,j] = Rstore[n,i,j]
                I[i,j] = Istore[n,i,j]

        Rmult = MultRstore[n]
        Imult = MultIstore[n]

        if sorteigenvalues=="MinDifference":
            # Find min combination

            MinMax_numerator=1e10

        elif sorteigenvalues=="MaxDifference":
            # Find max combination
            MinMax_numerator=0.

        for m in range(6):
            mysum=0.
            ind=Perm[m,:]
            puI=np.zeros(3,dtype=np.longdouble)
            for i in range(3):
                puI[i]=uI[ind[i]-1]
            #for i in range(3):
            #    mysum = mysum+ (uR[i]-puI[i])**2
            # Update critera to instead minimise or maximise the denominator in F!
            uRI,VRI = np.linalg.eig((R-I).astype(dtype=float))
            uRI=np.sort(uRI)
            uRImuUI=np.sort(uR-puI)

            numerator=0.
            for i in range(3):
                y=uRImuUI[i]
                x=uRI[i]
                # Rewrite (x^2-y^2) as (x+y)*(x-y)
                numerator=numerator+(x+y)*(x-y)
            numerator=abs(numerator)


            check = False
            if sorteigenvalues=="MinDifference" and numerator < MinMax_numerator:
                check = True
            elif sorteigenvalues=="MaxDifference" and numerator > MinMax_numerator:
                check = True
            if check==True:
                MinMax_numerator = numerator

        #
                #uRopt=np.copy(uR)
                #uIopt =np.copy(puI)
                for i in range(3):
                    SortedURstore[n,i]=uR[i]
                    SortedUIstore[n,i]=puI[i]

            # For the classifier we don't need to optimise the eigenvectors
            # We are only using the ordered eigenvalues so comment out this section
            ##########################################################
##                thetaopt=1e10
##                Kopt=np.zeros((3,3))
##                for k in range(8):
##                    QIordsign=np.zeros((3,3))
##                    for j in range(3):
##                        QIordsign[:,j] = sign[k,j]*QI[:,ind[j]-1]
##                    for kk in range(8):
##                        QRordsign=np.zeros((3,3))
##                        for j in range(3):
##        #             # Only the ordering of the columns of QI has been changed, but
##        #             # we can change the signs of the columns of QR and QI
##                            QRordsign[:,j] = sign[kk,j]*QR[:,j]
##        #
##        #
##                    if np.linalg.det(np.transpose(QRordsign)@QIordsign)> 0:
##        #         # Only do for valid rotation matrices with det(R) + ve (=1)
##                        theta, K, Tvec= Rodrigues(QRordsign, QIordsign)
##                        if theta < thetaopt:
##                            thetaopt=theta
##                            Kopt=np.copy(K)
##                            QRopt = np.copy(QRordsign)
##                            QIopt = np.copy(QIordsign)
##       
##            for j in range(3):
##                SortedQRstore[n,i,j] = QRopt[i,j]
##                SortedQIstore[n,i,j] = QIopt[i,j]
##                SortedKstore[n,i,j] = Kopt[i,j]
        

#####################################################################################################

    return SortedMultRstore, SortedMultIstore, SortedURstore, SortedUIstore, SortedQRstore, SortedQIstore, SortedKstore
