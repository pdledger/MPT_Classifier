import numpy as np

def Simple_UQ_bootstrap(k,probs,Number_Of_Classes,ProbabilitiesUpPL,ProbabilitiesLowPL,UQPL):
    # Also compute uq based nstand standard deviations
    nstand =  2

    row,col = np.shape(probs)
    uq = np.zeros(row)
    plow = np.zeros((row,Number_Of_Classes))
    pup = np.zeros((row,Number_Of_Classes))
    # Based on https://doi.org/10.2352/ISSN.2470-1173.2019.11.IPAS-264
    # pg 264-3
    for i in range(row):
        mean = 0.
        for j in range(Number_Of_Classes):
            mean = mean + probs[i,j]
        mean = mean / float(Number_Of_Classes)
        #print(mean,col)
        for j in range(Number_Of_Classes):
            uq[i] = uq[i] + (probs[i,j]-mean)**2
        uq[i] = uq[i] / float(Number_Of_Classes)
        uq[i] = 1./float(Number_Of_Classes) - 1./float(Number_Of_Classes)**2 - uq[i]
        if uq[i] < 0.:
            print('Negative UQ!')
        # This is average vairance.
        # We want to do +/- nstand standard deviations so
        uq[i] = nstand*np.sqrt(uq[i])
        for j in range(Number_Of_Classes):
            pup[i,j] = np.min((probs[i,j] + uq[i],1))
            plow[i,j] = np.max((probs[i,j] - uq[i],0))
            #print(pup[i,:],plow[i,:])
            #print(uq[i])
            #time.sleep(10)
            #time.sleep(100)
    if k ==0:
        ProbabilitiesUpPL = pup
        ProbabilitiesLowPL = plow
        UQPL = uq
    else:
        ProbabilitiesUpPL = np.vstack((ProbabilitiesUpPL,pup))
        ProbabilitiesLowPL = np.vstack((ProbabilitiesLowPL,plow))
        UQPL = np.append(UQPL,uq)

#print('uqs',min(UQPL),max(UQPL),np.amax(ProbabilitiesUpPL),np.amin(ProbabilitiesLowPL))
#time.sleep(10)
    return ProbabilitiesUpPL,ProbabilitiesLowPL,UQPL
