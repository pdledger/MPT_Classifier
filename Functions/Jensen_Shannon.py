import time
import scipy
import matplotlib.pyplot as plt
import numpy as np


def Jensen_Shannon(Vary_alpha_beta,AlphaList,BetaList,Number_Of_Classes,reordered_names,DataSet_Name,Testing_noise,Model,Savename):
    # Number of enteries in alphalist
    N=len(AlphaList)
    NSamp=100
    # each has the form = { "Class": Pdfclass, "Post_prob": Pdfdata}

    # We have two nested lists
    # We have list considering different alpha, beta combinations. Each item is also a list.
    # These (sub) lists have items which each contain a Pandas dataframe.
    # Each pandas dataframe comprises of the predictions obtained for some specified
    # measured data.
    plt.figure()

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
            hist1, edges1=np.histogram(Data[0,:],range=(0.,1.))
            hist2, edges2=np.histogram(Data[1,:],range=(0.,1.))
            hist3, edges3=np.histogram(Data[2,:],range=(0.,1.))
            # Compare alpha=beta =1 and alpha = beta 0.1
            res=jensen_shannon_distance(hist1/len(Data[0,:]), hist2/len(Data[1,:]))#scipy.stats.kstest(Data[0,:],Data[1,:])
            res2 = scipy.spatial.distance.jensenshannon(hist1/len(Data[0,:]),hist2/len(Data[1,:]))
            Comp_Pstat[k,0] = res

        plt.plot(np.linspace(0,Number_Of_Classes-1,Number_Of_Classes),Comp_Pstat[:,0],label="Correct Class="+str(myk))
    plt.xlabel("Class")
    plt.ylabel("Jensen-Shannon Distance")
    plt.ylim(0,1)
    #plt.yticks(reordered_names, rotation='vertical')
    plt.legend()
    if type(Testing_noise) == bool:
        plt.savefig('Results/'+DataSet_Name+'/Noiseless/'+Model+'/'+Savename+'/Jensen_Shannon_alp_bet_1_vs_alp_bet_0_1.pdf')
    else:
        plt.savefig('Results/'+DataSet_Name+'/Noise_'+str(Testing_noise)+'/'+Model+'/'+Savename+'/Jensen_Shannon_alp_bet_1_vs_alp_bet_0_1.pdf')
    plt.close()
            

    plt.figure()

    # Loop over each measured data (true class)
    for myk in range(Number_Of_Classes):
        # Loop over classification classes
        Comp_Pstat=np.zeros((Number_Of_Classes,2))
        for k in range(Number_Of_Classes):
            key = reordered_names[k]
            # loop over different alpha, beta combinations.
            Data=np.zeros((N,NSamp))#500))#NSamp))
            for n in np.array((0,2)):#(N):
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
                
                
                #time.sleep(10)
            # Now that the data has been obtained we want to compare them
            # Convert to a histogram with relative frequencies
            hist1, edges1=np.histogram(Data[0,:],range=(0.,1.))
            hist3, edges3=np.histogram(Data[2,:],range=(0.,1.))

            # Compare alpha=beta =1 and alpha = beta 0.01
            res=jensen_shannon_distance(hist1/len(Data[0,:]), hist3/len(Data[2,:]))#scipy.stats.kstest(Data[0,:],Data[1,:])
            res2 = scipy.spatial.distance.jensenshannon(hist1/len(Data[0,:]),hist3/len(Data[2,:]))
            Comp_Pstat[k,1] = res

            #print(res)
            #Comp_Pstat[k,0] = res.pvalue
            #print(res)
            #res=scipy.stats.kstest(Data[0,:],Data[2,:])
            #Comp_Pstat[k,1] = res.pvalue
        #print(Comp_Pstat[:,0])
        plt.plot(np.linspace(0,Number_Of_Classes-1,Number_Of_Classes),Comp_Pstat[:,1],label="Correct Class="+str(myk))
    plt.xlabel("Class")
    plt.ylabel("Jensen-Shannon Distance")
    plt.ylim(0,1)
    #plt.yticks(reordered_names, rotation='vertical')
    plt.legend()
    if type(Testing_noise) == bool:
        plt.savefig('Results/'+DataSet_Name+'/Noiseless/'+Model+'/'+Savename+'/Jensen_Shannon_alp_bet_1_vs_alp_bet_0_01.pdf')
    else:
        plt.savefig('Results/'+DataSet_Name+'/Noise_'+str(Testing_noise)+'/'+Model+'/'+Savename+'/Jensen_Shannon_alp_bet_1_vs_alp_bet_0_01.pdf')
    plt.close()
            

    
def jensen_shannon_distance(p, q):
#    """
#    method to compute the Jenson-Shannon Distance 
#    between two probability distributions
#    """

    # convert the vectors into numpy arrays in case that they aren't
    p = np.array(p)
    q = np.array(q)

    # calculate m
    m = (p + q) / 2

    # compute Jensen Shannon Divergence
    divergence = (scipy.stats.entropy(p, m) + scipy.stats.entropy(q, m)) / 2

    # compute the Jensen Shannon Distance
    print(divergence)
    distance = np.sqrt(divergence)

    return distance                  
           
