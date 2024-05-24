import numpy as np
def Commeasure(sorteigenvalues,SortedURstore, SortedUIstore, SortedQRstore, SortedQIstore, SortedKstore, Rstore,Istore, Frequencies):
    N=len(Frequencies)
    Comexactconst = np.zeros(N)
    #Fapproxconst = np.zeros(N)
    Comapproxconst_min = np.zeros(N)
    Comapproxconst_max = np.zeros(N)
    den_const= np.zeros(N)
    # First compute Commutator measure using the exact constant
    # For the classifier we don't need the exact constant so comment out this part
    ########################################################################
##    for n in range(N):
##        QR = np.zeros((3,3))
##        QI = np.zeros((3,3))
##        R=np.zeros((3,3))
##        I=np.zeros((3,3))
##        uR = np.zeros(3)
##        uI = np.zeros(3)
##        K = np.zeros((3,3))
##        for i in range(3):
##            uR[i] = SortedURstore[n,i]
##            uI[i] = SortedUIstore[n,i]
##            for j in range(3):
##                QR[i,j] = SortedQRstore[n,i,j]
##                QI[i,j] = SortedQIstore[n,i,j]
##                K[i,j] = SortedKstore[n,i,j]
##                R[i,j] = Rstore[n,i,j]
##                I[i,j] = Istore[n,i,j]
##        # Compute commutator
##        Z = R@I - I@R
##
##        normalisation= 2*np.trace(np.diag(uR)**2@K@K@np.diag(uI)**2 )
##        normalisation -= 2*np.trace(np.diag(uR)**2@K@np.diag(uI)**2@K)
##        normalisation -= 4*np.trace(K@np.diag(uR)@np.diag(uI)@K@np.diag(uR)@np.diag(uI))
##        normalisation -= 4*np.trace(K@K@np.diag(uR)**2@np.diag(uI)**2)
##        normalisation += 4*np.trace(K@np.diag(uR)@K@np.diag(uR)*np.diag(uI)**2)
##        normalisation += 4*np.trace(K@np.diag(uI)@K@np.diag(uI)*np.diag(uR)**2)
##
##        Comexactconst[n] = np.abs(np.linalg.norm(Z,ord='fro')**2 ) / np.abs(normalisation)
##    Comexactconst= np.sqrt(Comexactconst)
    ##############################################################


    # Next compute F using the apprx constant without K, QR or QI
    for n in range(N):
        R=np.zeros((3,3))
        I=np.zeros((3,3))
        uR = np.zeros(3)
        uI = np.zeros(3)
        for i in range(3):
            uR[i] = SortedURstore[n,i]
            uI[i] = SortedUIstore[n,i]
            for j in range(3):
                R[i,j] = Rstore[n,i,j]
                I[i,j] = Istore[n,i,j]
        # Compute commutator
        Z = R@I - I@R

        #normalisationapprox=0
        #for j in range(3):
        #    normalisationapprox-=2*(uI[j]*uR[j])/np.sqrt(3)
        #normalisationapprox+=((uI[1]*uR[0])+(uI[2]*uR[0])+(uI[0]*uR[1])+(uI[2]*uR[1])+(uI[0]*uR[2])+(uI[1]*uR[2]))/np.sqrt(3)
        evlist = np.zeros(3)
        evlist[0]=2*(uR[1]-uR[2])**2*(uI[1]-uI[2])**2
        evlist[1]=2*(uR[0]-uR[2])**2*(uI[0]-uI[2])**2
        evlist[2]=2*(uR[0]-uR[1])**2*(uI[0]-uI[1])**2

        normalisation_min = np.min(np.abs(evlist))
        normalisation_max = np.max(np.abs(evlist))
        Comapproxconst_min[n] = np.min([np.abs(np.linalg.norm(Z,ord='fro')**2 ) / normalisation_min,
            np.abs(np.linalg.norm(Z,ord='fro')**2 ) / normalisation_max])
        Comapproxconst_max[n] = np.max([np.abs(np.linalg.norm(Z,ord='fro')**2 ) / normalisation_min,
            np.abs(np.linalg.norm(Z,ord='fro')**2 ) / normalisation_max])

        #Comapproxconst_min[n] = np.abs(np.linalg.norm(Z,ord='fro')**2 ) / np.abs(normalisation_min)
        #Comapproxconst_max[n] = np.abs(np.linalg.norm(Z,ord='fro')**2 ) / np.abs(normalisation_max)
        den_const[n]=np.min([np.sqrt(np.abs(normalisation_min)),np.sqrt(np.abs(normalisation_max))])
#        Fapproxconst[n] = np.abs(np.linalg.norm(R-I,ord='fro')**2 - diffeig) / np.abs(normalisationapprox)
#    Fapproxconst= np.sqrt(Fapproxconst)
    Comapproxconst_min=np.sqrt(Comapproxconst_min)
    Comapproxconst_max=np.sqrt(Comapproxconst_max)



    return Comexactconst,Comapproxconst_min,Comapproxconst_max, den_const
