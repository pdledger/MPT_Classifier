import numpy as np
import scipy.linalg
import time

from Rodrigues import *
from StableAngle import *
from CheckOrdering import *

def MinMaxthetafromQRQI(Frequencies,QRstore,QIstore,URstore, UIstore,MultRstore,MultIstore,FixEvecs):

    # For each freuqncy use multiplicties to determine method for
    # obtaining angles
    N = len(Frequencies)
    MinAnglestore = np.zeros(N, dtype=np.longdouble)
    MaxAnglestore = np.zeros(N, dtype=np.longdouble)
    dFMinAnglestoreRI = np.zeros(N, dtype=np.longdouble)
    dFMaxAnglestoreRI = np.zeros(N, dtype=np.longdouble)
    SortedURstore =np.zeros((N,3))
    SortedUIstore =np.zeros((N,3))
    SortedQRstore = np.zeros((N,3,3))
    SortedQIstore = np.zeros((N,3,3))
    SortedKstore = np.zeros((N,3,3))

    for n in range(N):
        QR = np.zeros((3,3), dtype=np.longdouble)
        QI = np.zeros((3,3), dtype=np.longdouble)
        uR =np.zeros(3, dtype=np.longdouble)
        uI = np.zeros(3, dtype=np.longdouble)
        for i in range(3):
            uR[i]=URstore[n,i]
            uI[i]=UIstore[n,i]
            for j in range(3):
                QR[i,j] = QRstore[n,i,j]
                QI[i,j] = QIstore[n,i,j]
        Rmult = MultRstore[n]
        Imult = MultIstore[n]
        #print(Rmult,Imult)
        if Rmult != Imult :
            print("error different multiplicties for R and I",Rmult,Imult)
            Tol=1e-4*np.min(np.abs(uR))
            #for i in range(3):
            #    print(uR[i],np.linalg.matrix_rank(R-uR[i]*np.eye(3),tol=Tol))
            #Tol=1e-4*np.min(np.abs(uI))
            #for i in range(3):
            #    print(uI[i],np.linalg.matrix_rank(I-uI[i]*np.eye(3),tol=Tol))

            if Rmult==2 or Imult==2:
                Rmult=2
                Imult=2
            else:
                Rmult=1
                Imult=1


        if Rmult==3:
            # This is a case where all 3 eigenvalues are the same. So we know that
            QR=QI
            max_angle, K, tvec = Rodrigues(QR,QI)

            min_angle, K, tvec = Rodrigues(QR,QI)
            dF_min = dFmetric(QR,QI)
            dF_max = dFmetric(QR,QI)

        if Rmult==2:
            # Do a check for the first eigenvector
            angle = np.zeros(6)
            #print(QR,QI,uR,uI)
            #time.sleep(10)

            #QR, QI, uR = CheckOrdering(QR,QI,uR,1)
            # Arrange for a maximal angle
            if FixEvecs=="Yes":
                if np.sign(QR[:,0]@QI[:,0])> 0:
                    QR[:,0] = - QR[:,0]
                QR[:,1:3] = QI[:,1:3]
            else:
                QR, QI, uR, uI = CheckOrdering(QR,QI,uR,uI,1)
            max_angle, K, tvec = Rodrigues(QR,QI)
            dF_max = dFmetric(QR,QI)


            # Arrange for a mimimal angle
            if FixEvecs=="Yes":
                if np.sign(QR[:,0]@QI[:,0])< 0:
                    QR[:,0] = - QR[:,0]
                QR[:,1:3] = QI[:,1:3]
            else:
                QR, QI, uR, uI = CheckOrdering(QR,QI,uR,uI,0)
            #min_angle=np.arccos(QR[:,0]@QI[:,0])
            #K=np.zeros((3,3))
            #K[1,2]=-1
            #K[2,1]=1
            # Find the combinations of QR and QI that lead to a minimal angle
            #QR, QI, uR = CheckOrdering(QR,QI,uR,0)
            min_angle, K, tvec = Rodrigues(QR,QI)
            #print(min_angle)
            dF_min = dFmetric(QR,QI)
            #print(min_angle)
            #time.sleep(10)



        if Rmult==1:
            # Find the combinations of QR and QI that lead to a minimal angle
            QR, QI, uR, uI = CheckOrdering(QR,QI,uR,uI,1)
            max_angle, K, tvec = Rodrigues(QR,QI)
            dF_max = dFmetric(QR,QI)
            QR, QI, uR, uI = CheckOrdering(QR,QI,uR,uI,0)
            min_angle, K, tvec = Rodrigues(QR,QI)
            dF_min = dFmetric(QR,QI)


        MinAnglestore[n] = min_angle
        MaxAnglestore[n] = max_angle
        dFMinAnglestoreRI[n] = dF_min
        dFMaxAnglestoreRI[n] = dF_max
        for i in range(3):
            SortedURstore[n,i]=uR[i]
            SortedUIstore[n,i]=uI[i]

            for j in range(3):
                SortedQRstore[n,i,j] = QR[i,j]
                SortedQIstore[n,i,j] = QI[i,j]
                SortedKstore[n,i,j] = K[i,j]



    return MinAnglestore, MaxAnglestore, dFMinAnglestoreRI, dFMaxAnglestoreRI, \
    SortedURstore, SortedUIstore, SortedQRstore, SortedQIstore, SortedKstore

def dFmetric(QR,QI):
    return np.sqrt(np.trace((QR-QI)@np.transpose(QR-QI)))
