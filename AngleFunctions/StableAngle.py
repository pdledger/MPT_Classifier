import numpy as np

def StableAngle(Q):
    # Use alternative formulation for small angles
    # From Taylor's series Q = I + theta K, ||K||_F =2 and so
    theta=np.linalg.norm(Q-np.eye(3),ord='fro')/np.sqrt(2.)
    #print(theta)
    if theta > 0.2:
        # standard computation
        theta = np.arccos((np.trace(Q)-1.)/2.)
        #print(theta)
    return np.abs(theta)
