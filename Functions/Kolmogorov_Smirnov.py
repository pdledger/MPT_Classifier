import time
import scipy
import matplotlib.pyplot as plt
import numpy as np


def Kolmogorov_Smirnov(Vary_alpha_beta,AlphaList,BetaList,Number_Of_Classes,reordered_names):
    # Number of enteries in alphalist
    N=len(AlphaList)
    NSamp=100
    # each has the form = { "Class": Pdfclass, "Post_prob": Pdfdata}

    # We have two nested lists
    # We have list considering different alpha, beta combinations. Each item is also a list.
    # These (sub) lists have items which each contain a Pandas dataframe.
    # Each pandas dataframe comprises of the predictions obtained for some specified
    # measured data.

    # Loop over each measured data (true class)
    for myk in range(Number_Of_Classes):
        # Loop over classification classes
        Comp_Pstat=np.zeros((Number_Of_Classes,2))
        for k in range(Number_Of_Classes):
            key = reordered_names[k]
            # loop over different alpha, beta combinations.
            Data=np.zeros((N,NSamp))#500))#NSamp))
            for n in range(2):#(N):
                # Pick an alpha, beta combination
                df_alpha_beta = Vary_alpha_beta[n]
                # Pick true data class
                df = df_alpha_beta[myk]

                # we are interested in those Post_prob 's with Class being the key
                series = df.loc[df["Class"] == key, "Post_prob"]
                #print(series)
                #print(series.to_numpy())
                numpy_array = series.to_numpy()
                # fit data to normal for perform test
                mu,std = scipy.stats.norm.fit(numpy_array)
                Data[n,:] = numpy_array #scipy.stats.norm.rvs(loc=mu,scale=std,size=500)
                #print(np.shape(numpy_array))
                
                #time.sleep(10)
            # Now that the data has been obtained we want to compare them
            # Convert to a histogram with relative frequencies
            res=scipy.stats.kstest(Data[0,:],Data[1,:])
            Comp_Pstat[k,0] = res.pvalue
            #print(res)
            #res=scipy.stats.kstest(Data[0,:],Data[2,:])
            #Comp_Pstat[k,1] = res.pvalue
        print(Comp_Pstat[:,0])
        plt.figure()
        plt.plot(reordered_names,1-Comp_Pstat[:,0],label="alpha=beta=1 vs alpha=beta=0.1"))
        #plt.plot(reordered_names,0.95*np.ones(Number_Of_Classes),label="Cannot reject Null-hypothesis below")                  
        plt.xlabel("Class")
        plt.ylabel("1-p")
        yticks(reordered_names, rotation='vertical')
        plt.legend()
        plt.show()

    
