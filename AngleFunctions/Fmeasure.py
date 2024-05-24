import numpy as np
import time
def Fmeasure(sorteigenvalues,SortedURstore, SortedUIstore, SortedQRstore, SortedQIstore, SortedKstore, Rstore,Istore, Frequencies):
    N=len(Frequencies)
    Fexactconst = np.zeros(N,dtype=np.longdouble)
    Fexactconst3 = np.zeros(N,dtype=np.longdouble)
    Fexactconst4 = np.zeros(N,dtype=np.longdouble)


    #Fapproxconst = np.zeros(N)
    Fapproxconst_min = np.zeros(N,dtype=np.longdouble)
    Fapproxconst_max = np.zeros(N,dtype=np.longdouble)
    den_const=np.zeros(N)
    # First compute F using the exact constant
    # For the classifier we don't need the exact constant so comment this part out
    ##############################################################################
##    for n in range(N):
##        QR = np.zeros((3,3),dtype=np.longdouble )
##        QI = np.zeros((3,3),dtype=np.longdouble)
##        R=np.zeros((3,3),dtype=np.longdouble)
##        I=np.zeros((3,3),dtype=np.longdouble)
##        uR = np.zeros(3,dtype=np.longdouble)
##        uI = np.zeros(3,dtype=np.longdouble)
##        K = np.zeros((3,3),dtype=np.longdouble)
##        for i in range(3):
##            uR[i] = SortedURstore[n,i]
##            uI[i] = SortedUIstore[n,i]
##            for j in range(3):
##                QR[i,j] = SortedQRstore[n,i,j]
##                QI[i,j] = SortedQIstore[n,i,j]
##                K[i,j] = SortedKstore[n,i,j]
##                R[i,j] = Rstore[n,i,j]
##                I[i,j] = Istore[n,i,j]
##        diffeig=0.
##        for i in range(3):
##            diffeig+=(uR[i]-uI[i])**2
##        normalisation =np.abs( np.trace(K@K@np.diag((uR))@np.diag((uI))))- np.abs(np.trace(K@np.diag((uR))@K@np.diag((uI))))
##        Fexactconst[n] = np.abs(np.linalg.norm(R-I,ord='fro')**2 - diffeig) / np.abs(normalisation)
##        #normalisation =np.abs( np.trace(K@K@np.diag((uR))@K@np.diag((uI))))- np.abs(np.trace(K@K@np.diag((uI))@K@np.diag((uR))))
##        #Fexactconst3[n] = 2*np.abs(np.linalg.norm(R-I,ord='fro')**2 - diffeig) / np.abs(normalisation)
##        #normalisation =np.abs( np.trace(K@K@np.diag((uR))@K@K@np.diag((uI))))
##        #Fexactconst4[n] = 4*np.abs(np.linalg.norm(R-I,ord='fro')**2 - diffeig) / np.abs(normalisation)
##
##    Fexactconst= np.sqrt(Fexactconst)
##    #Fexactconst3= (Fexactconst3)**(1/3)
    #Fexactconst4= (Fexactconst4)**(1/4)
    ###########################################################################################################




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
        diffeig=0.
        #print(Frequencies[n],uR,uI)
        for i in range(3):
            diffeig=diffeig+(uR[i]-uI[i])**2
        diffeig=diffeig

        #print(Frequencies[n],"Combination of uR",uR)

        #print(Frequencies[n],"Combination of uI",uI)


        
        evlist = np.zeros(3)

        evlist[0]= - (uI[1]-uI[2])*(uR[1]-uR[2])
        evlist[1]= - (uI[0]-uI[2])*(uR[0]-uR[2])
        evlist[2]= - (uI[0]-uI[1])*(uR[0]-uR[1])


        normalisation_min = np.min(np.abs(evlist))
        normalisation_max = np.max(np.abs(evlist))
        
        Tol=1e-6
        
        # We can rewrite ||R -I||^2 = sum_i=1^3 lambda_i^2(R-I)
        uRI,VRI = np.linalg.eig((R-I).astype(dtype=float))
        uRI=np.sort(uRI)
        uRImuUI=np.sort(uR-uI)

        numerator=0.
        for i in range(3):
            y=uRImuUI[i]
            x=uRI[i]
            # Rewrite (x^2-y^2) as (x+y)*(x-y)
            numerator=numerator+(x+y)*(x-y)
        

        Calc1= np.abs( numerator /normalisation_min)
        Calc2= np.abs( numerator /normalisation_max)
        
        Fapproxconst_min[n] = np.min([ Calc1,
                                        Calc2])
        Fapproxconst_max[n] = np.max([ Calc1,
                                Calc2])

        den_const[n]=np.min([np.sqrt(np.abs(normalisation_min)),np.sqrt(np.abs(normalisation_max))])

    Fapproxconst_min=(Fapproxconst_min)**0.5
    Fapproxconst_max=(Fapproxconst_max)**0.5

    return Fexactconst,Fapproxconst_min,Fapproxconst_max,den_const
    #Fapproxconst
