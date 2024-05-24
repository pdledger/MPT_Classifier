import numpy as np

from StableAngle import *

# Function for looping over different possible orderings of columns of QR for fixed QI
# to work out the ordering that leads to minimal or maximal angle
def CheckOrdering(QR,QI,uR,minmax):


    option=np.array([[ 0,1,2], [0,2,1], [1,0,2], [1, 2, 0], [2, 0, 1 ], [2, 1, 0], \
                     [ 0,1,2], [0,2,1], [1,0,2], [1, 2, 0], [2, 0, 1 ], [2, 1, 0], \
                     [ 0,1,2], [0,2,1], [1,0,2], [1, 2, 0], [2, 0, 1 ], [2, 1, 0], \
                     [ 0,1,2], [0,2,1], [1,0,2], [1, 2, 0], [2, 0, 1 ], [2, 1, 0], \
                     [ 0,1,2], [0,2,1], [1,0,2], [1, 2, 0], [2, 0, 1 ], [2, 1, 0], \
                     [ 0,1,2], [0,2,1], [1,0,2], [1, 2, 0], [2, 0, 1 ], [2, 1, 0], \
                     [ 0,1,2], [0,2,1], [1,0,2], [1, 2, 0], [2, 0, 1 ], [2, 1, 0], \
                     [ 0,1,2], [0,2,1], [1,0,2], [1, 2, 0], [2, 0, 1 ], [2, 1, 0]])
    signs=[]
    signs=np.array([[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1], \
     [-1,1,1],[-1,1,1],[-1,1,1],[-1,1,1],[-1,1,1],[-1,1,1],[-1,1,1], \
     [-1,-1,1],[-1,-1,1],[-1,-1,1],[-1,-1,1],[-1,-1,1],[-1,-1,1],[-1,-1,1], \
     [-1,-1,-1],[-1,-1,-1],[-1,-1,-1],[-1,-1,-1],[-1,-1,-1],[-1,-1,-1],[-1,-1,-1], \
     [-1,1,-1],[-1,1,-1],[-1,1,-1],[-1,1,-1],[-1,1,-1],[-1,1,-1],[-1,1,-1], \
     [1,-1,-1],[1,-1,-1],[1,-1,-1],[1,-1,-1],[1,-1,-1],[1,-1,-1],[1,-1,-1], \
     [1,-1,1],[1,-1,1],[1,-1,1],[1,-1,1],[1,-1,1],[1,-1,1],[1,-1,1], \
     [1,1,-1],[1,1,-1],[1,1,-1],[1,1,-1],[1,1,-1],[1,1,-1],[1,1,-1]])

    if minmax==0:
        theta=1e2
    elif minmax==1:
        theta=0.


    uRorg=np.copy(uR)
    QRorg=np.copy(QR)
    uIorg=np.copy(uI)
    QIorg=np.copy(QI)

    for n in range(48):
        ind= option[n,:]
        uR=np.copy(uRorg)
        QR=np.copy(QRorg)
        uR = uR[ind]
        QR = QR[:,ind]
        for i in range(3):
            QR[:,i]=signs[n,i]*QR[:,i]

        for m in range(48):
            ind= option[m,:]
            uI=np.copy(uIorg)
            QI=np.copy(QIorg)
            uI = uI[ind]
            QI = QI[:,ind]


            Q = np.transpose(QR)@QI
            Q = Q /np.linalg.norm(Q,ord=2)
            tol=0.01
            if np.linalg.det(Q)> 0:
                #LogQ = scipy.linalg.logm(Q)
                # determine the angle for the current combination
                ntheta = StableAngle(Q)
                if minmax==0 and ntheta < theta:
                    theta=ntheta
                    noutn=n
                    noutm=m
                elif minmax==1 and ntheta > theta and ntheta-tol < np.pi/2 : #and np.abs(ntheta- np.pi/2) > tol and
                    theta=ntheta
                    noutn=n
                    noutm=m


    # For the chosen indices output uR,QR
    # Remember ind contains a LIST of indices.
    try: noutn
    except NameError: noutn = None
    if noutn != None:
        ind= option[noutn,:]
        uR=np.copy(uRorg)
        QR=np.copy(QRorg)

        uR = uR[ind]
        QR = QR[:,ind]
        for i in range(3):
            QR[:,i]=signs[noutn,i]*QR[:,i]

        ind= option[noutm,:]
        uI=np.copy(uIorg)
        QI=np.copy(QIorg)

        uI = uR[ind]
        QI = QR[:,ind]
        for i in range(3):
            QI[:,i]=signs[noutm,i]*QI[:,i]

    return QR, QI, uR, uI
