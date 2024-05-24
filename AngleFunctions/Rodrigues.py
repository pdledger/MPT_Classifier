
import numpy as np
import scipy.linalg

from StableAngle import *

def Rodrigues(QR, QI):
    Q = np.transpose(QR)@QI
    #print(np.linalg.norm(Q,ord=2))
    Q = Q /np.linalg.norm(Q.astype(dtype=float),ord=2)
    # This will be a rotation matrix and so can be expressed as Q = exp(theta K)
    # Take the matrix logarithm to find K and theta
    LogQ = scipy.linalg.logm(Q)

    # We need to take when computing the angle (especially for small angles)
    theta = StableAngle(Q)

    if np.abs(np.sin(theta)) > 1e-3:
        K = LogQ / theta
        # This is the unit vector K
        Kvec = 1/(2*np.sin(theta))*np.array((Q[2,1]-Q[1,2],Q[0,2]-Q[2,0],Q[1,0]-Q[0,1]), dtype=np.longdouble)
        #Kvec[0]=-K[1,2]
        #Kvec[1]=K[0,2]
        #Kvec[2]=K[1,0]
    else:
        # angle not defined so just fix Kvec =[0,0,0]
        K = np.zeros((3,3), dtype=np.longdouble)
        Kvec=np.zeros(3, dtype=np.longdouble)

    # This is the Euler Angle-Axis description
    Tvec=theta*Kvec

    return theta, K, Tvec
