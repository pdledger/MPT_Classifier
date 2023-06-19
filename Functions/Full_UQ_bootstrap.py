import numpy as np
import statistics
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde

def Full_UQ_bootstrap(model,k,probs,Number_Of_Classes,ProbabilitiesUpPL,ProbabilitiesLowPL,UQPL,X_test_norm,ProbabilitiesPL,DataSet_Name,Model,Savename,Testing_noise,Y_test,PYCOL,reordered_names):

    if k != 0:
        disp("Warning attempting to do more than a single bootstrap iteration")
        disp("but Bayesian classification is selected - so only a single")
        disp("bootstrap iteration is expected")
        exit()
        
    # Also compute uq based nstand standard deviations
    nstand =  2

    N = np.shape(X_test_norm)[0] # Number of items
    D = np.shape(X_test_norm)[1] # Number of features
    # Previously on the MAP estimates were computed - now we go for the full
    # probability distribution
#    for n in range(N):
#        X_test_normn = X_test_norm(n,:)
    probs=np.array([],dtype=float)

    uq=np.zeros((N,Number_Of_Classes)) # nb different dimension to non-Bayesian we now have
                                    # a different UQ estimate for every class

    plow = np.zeros((N,Number_Of_Classes))
    pup = np.zeros((N,Number_Of_Classes))
    plow90 = np.zeros((N,Number_Of_Classes))
    pup90 = np.zeros((N,Number_Of_Classes))
    pmean = np.zeros((N,Number_Of_Classes))

    flagged = np.zeros(Number_Of_Classes)
                                    
    # Keep a record of first distribution (only) set 100 samples from the posterior
    nsamp=100
    postdist = np.zeros((Number_Of_Classes,nsamp))

    for myk in range(Number_Of_Classes):
        if myk==0:
            cols = ["Class"+str(myk)]
        else:
            cols.append("Class"+str(myk))

#   To keep things simple go thorugh an item at a time
    count=0
    for n in range(N):
        print("Item",n+1," of ",N)
        for myk in range(Number_Of_Classes):
            kclass = myk*np.ones(1).astype('float32')
            x = np.zeros((1,D),dtype='float32')
            x[0,:] = X_test_norm[n,:].astype('float32') 
            #print(np.shape(x),np.shape(kclass))
            if  flagged[int(round(Y_test[n]))]== 0:

                # Determine the distribution and associated UQ
                probk = model.prob(x,kclass,distribution=True, n=nsamp)
                # model.prob gives the MAP estimate for P(C=k| X_test_norm[n,:])
                # now as a probability distribution. We already have the MAP estimate (in probs)
                # - we could get the mean too, and std. deviation
                mean = statistics.mean(probk[0,:]) # don't store as we have the MAP estimate in  probs?
                std = statistics.stdev(probk[0,:])
                pmean[n,myk] = mean
      
                # This would construct a 95 % confidence interval
                #uq[n,myk]= 1.96*std/np.sqrt(nsamp)
                #pup[n,myk] =  np.minimum(mean + 1.96*std/np.sqrt(nsamp), 1.0)
                #plow[n,myk] =  np.maximum(mean - 1.96*std/np.sqrt(nsamp), 0.0)

                # instead we want a 95 % credible interval.
                # Take the samples, order them, remove the top and bottom 2.5% the remainimg 95% forms the credible interval
                sortedprobs = np.sort(probk[0,:])
               # print(np.shape(sortedprobs),sortedprobs)
                lowind =max(int(np.ceil(0.025*nsamp))-1,0)
                highind = max(0,nsamp-1-lowind)
                pup[n,myk] = sortedprobs[highind]
                plow[n,myk] = sortedprobs[lowind]
                uq[n,myk] = pup[n,myk]-plow[n,myk]
               #Repeat for 90% CI
                lowind =max(int(np.ceil(0.05*nsamp))-1,0)
                highind = max(0,nsamp-1-lowind)
                pup90[n,myk] = sortedprobs[highind]
                plow90[n,myk] = sortedprobs[lowind]
                # As an example store the posterioir probabilities distributions of the first item
                postdist[myk,:] = probk[0,:]

            else:
                # Just get the MAP estimates and save - don't do distribution and UQ - quicker.
                probk = model.prob(x,kclass,distribution=False)
                pmean[n,myk] = probk[0]
                postdist[myk,:] = probk[0]
        
               
            
            #if n == 0:
                # As an example store the posterioir probabilities distributions of the first item
            
        if  flagged[int(round(Y_test[n]))]== 0:
            count=count+1
            # Keep a record and only output posterior distribution once for each class - just to show examples
            flagged[int(round(Y_test[n]))]=1
                        
            df2 = pd.DataFrame(np.transpose(postdist),columns=cols)
            fig=sns.histplot(df2,multiple="layer",bins=nsamp,stat="count")
            plt.xlabel("Posterior P(C_k=Class| x )")

            
            if type(Testing_noise) == bool:
                plt.savefig('Results/'+DataSet_Name+'/Noiseless/'+Model+'/'+Savename+'/posthistdistribution'+str(count)+'.pdf')
            else:
                plt.savefig('Results/'+DataSet_Name+'/Noise_'+str(Testing_noise)+'/'+Model+'/'+Savename+'/posthistdistribution'+str(count)+'.pdf')
            plt.close()
            print("Completed posterior plots")
            del fig

##            fig, ax = plt.subplots()
##            for myk in range(Number_Of_Classes):
##                sns.kdeplot(postdist[myk,:].cumsum(), 
##                             bw_adjust=0.5,multiple="layer",common_grid=True,clip=(0,1),ax=ax,label="Class"+str(myk))
##            ax.legend()
##            plt.tight_layout()
##            plt.xlabel('P(C_k=Class| x )')
##            if type(Testing_noise) == bool:
##                plt.savefig('Results/'+DataSet_Name+'/Noiseless/'+Model+'/'+Savename+'/postkdedistribution'+str(count)+'.pdf')
##            else:
##                plt.savefig('Results/'+DataSet_Name+'/Noise_'+str(Testing_noise)+'/'+Model+'/'+Savename+'/poskdedistribution'+str(count)+'.pdf')
##            plt.close()
##            print("Completed posterior plots")
##            del fig

            #data = np.random.normal(10,3,100) # Generate Data
            try:
                for myk in range(Number_Of_Classes):
                    density = gaussian_kde(postdist[myk,:],bw_method="silverman")
                
                    x_vals = np.linspace(-1,2,200) # Specifying the limits of our data
                    density.covariance_factor = lambda : 0.1 #Smoothing parameter
     
                    density._compute_covariance()
                    plt.plot(x_vals,density(x_vals),label="Class"+str(myk))


                plt.legend()
                plt.xlabel("Posterioir P(C_k|x)")
                plt.ylabel("Density")
                plt.yscale("log")
                plt.ylim(1e-2,1e2)
                plt.xlim(0,1)
                #plt.show()
                if type(Testing_noise) == bool:
                    plt.savefig('Results/'+DataSet_Name+'/Noiseless/'+Model+'/'+Savename+'/postkdedistribution'+str(count)+'.pdf')
                else:
                    plt.savefig('Results/'+DataSet_Name+'/Noise_'+str(Testing_noise)+'/'+Model+'/'+Savename+'/poskdedistribution'+str(count)+'.pdf')
                plt.close()
            except:
                print("Could not generate a kde plot for",Model,Testing_noise)
##            print("Completed posterior plots")


                       
            # Plot out bar graphs showing output for the uq value associated with this output using 95 % CI
            labels = ['Probability','95% Credible Interval']
            fig, ax = plt.subplots()
            Bars = ax.bar(np.arange(Number_Of_Classes), pmean[n,:],color=[PYCOL[3] if np.argmax(pmean[n,:])==j else PYCOL[0] for j in range(Number_Of_Classes)], align='center', alpha=0.5, ecolor='black', capsize=10,label='Probability')

            # including upper limits
            for j in range(Number_Of_Classes):
                ax.plot([j,j],[plow[n,j],pup[n,j]],linestyle='-', marker='_',markersize=16,color='k')

            Legend_elements = [Patch(facecolor=PYCOL[0],alpha=0.6,label=r'Mean posterior probability $p(C_k|$data)'),
                               Patch(facecolor=PYCOL[3],alpha=0.6,label=r'Mean posterior probability max$(p(C_k|$data))'),
                               plt.Line2D([0],[0],color='k', marker='_', markersize=16 ,linestyle=' ', label='95% Credible Interval Interval')]
            ax.set_xticks(np.arange(Number_Of_Classes))
            ax.set_xticklabels(reordered_names, rotation='vertical')
            plt.subplots_adjust(bottom=0.25)
            plt.gca().get_xticklabels()[int(Y_test[n])].set_fontweight('bold')
            ax.yaxis.grid(True)
            plt.ylim(0,1.05)
            plt.xlabel(r'Classes $C_k$')
            plt.ylabel(r'Posterior probability $p(C_k|$data)')
            plt.legend(handles=Legend_elements)
            if type(Testing_noise) == bool:
                plt.savefig('Results/'+DataSet_Name+'/Noiseless/'+Model+'/'+Savename+'/credint95snapshotfigure'+str(count)+'.pdf')
            else:
                plt.savefig('Results/'+DataSet_Name+'/Noise_'+str(Testing_noise)+'/'+Model+'/'+Savename+'/credint95snapshotfigure'+str(count)+'.pdf')
            plt.close()

            # Plot out bar graphs showing output for the uq value associated with this output using 90 % CI
            labels = ['Probability','90% Credible Interval']
            fig, ax = plt.subplots()
            Bars = ax.bar(np.arange(Number_Of_Classes), pmean[n,:],color=[PYCOL[3] if np.argmax(pmean[n,:])==j else PYCOL[0] for j in range(Number_Of_Classes)], align='center', alpha=0.5, ecolor='black', capsize=10,label='Probability')

            # including upper limits
            for j in range(Number_Of_Classes):
                ax.plot([j,j],[plow90[n,j],pup90[n,j]],linestyle='-', marker='_',markersize=16,color='k')

            Legend_elements = [Patch(facecolor=PYCOL[0],alpha=0.6,label=r'Mean posterior probability $p(C_k|$data)'),
                               Patch(facecolor=PYCOL[3],alpha=0.6,label=r'Mean posterior probability max$(p(C_k|$data))'),
                               plt.Line2D([0],[0],color='k', marker='_', markersize=16 ,linestyle=' ', label='90% Credible Interval')]
            ax.set_xticks(np.arange(Number_Of_Classes))
            ax.set_xticklabels(reordered_names, rotation='vertical')
            plt.subplots_adjust(bottom=0.25)
            plt.gca().get_xticklabels()[int(Y_test[n])].set_fontweight('bold')
            ax.yaxis.grid(True)
            plt.ylim(0,1.05)
            plt.xlabel(r'Classes $C_k$')
            plt.ylabel(r'Posterior probability $p(C_k|$data)')
            plt.legend(handles=Legend_elements)
            if type(Testing_noise) == bool:
                plt.savefig('Results/'+DataSet_Name+'/Noiseless/'+Model+'/'+Savename+'/credint90snapshotfigure'+str(count)+'.pdf')
            else:
                plt.savefig('Results/'+DataSet_Name+'/Noise_'+str(Testing_noise)+'/'+Model+'/'+Savename+'/credint90snapshotfigure'+str(count)+'.pdf')
            plt.close()

##            # Plot out bar graphs showing output for the uq value associated with this output
##            xb = np.arange(Number_Of_Classes)
##            print(xb)
##            wth=0.25
##            mult=0
##            yd={
##                'Mean': (pmean[n,:]),
##                'Mean -2*std': (plow[n,:]),
##                'Mean + 2*std': (pup[n,:])
##                }
##            #labels = ['Probability','UQ min']
##            fig, ax = plt.subplots(layout='constrained')
##            for attribute, value in yd.items():
##                    offset = wth * mult
##                    print(attribute,value)
##                    rects = ax.bar(xb + offset, value, wth, label=attribute)
##                    #ax.bar_label(rects, padding=3)
##                    mult += 1

                
            #Bars = ax.bar(np.arange(Number_Of_Classes), pmean[n,:],color=[PYCOL[3] if np.argmax(pmean[n,:])==j else PYCOL[0] for j in range(Number_Of_Classes)], align='center', alpha=0.5, ecolor='black', capsize=10,label='Probability')
        
##            Bars1 = ax.bar(np.arange(Number_Of_Classes)-0.1, pmean[n,:],color=PYCOL[0], width=0.2, ecolor='black', capsize=10,label='Mean posterior probability $p(C_k|$data)')
##            Bars2 = ax.bar(np.arange(Number_Of_Classes), plow[n,:],color=PYCOL[1], align='center', alpha=0.5, ecolor='black', capsize=10,label='Mean Posterior probability $(p(C_k|$data)) - 2* Std. Dev.')
##            Bars3 = ax.bar(np.arange(Number_Of_Classes)+0.1, pup[n,:],color=PYCOL[2], align='center', alpha=0.5, ecolor='black', capsize=10,label='Mean Posterior probability $(p(C_k|$data)) - 2* Std. Dev.')
##
##
##            # including upper limits
###            for j in range(Number_Of_Classes):
###                ax.plot([j,j],[plow[n,j],pup[n,j]],linestyle='-', marker='_',markersize=16,color='k')
##
###            Legend_elements = [Patch(facecolor=PYCOL[0],alpha=0.6,label=r'Mean posterior probability $p(C_k|$data)'),
###                               Patch(facecolor=PYCOL[2],alpha=0.6,label=r'Mean Posterior probability $(p(C_k|$data)) - 2* Std. Dev.'),
###                               Patch(facecolor=PYCOL[1],alpha=0.6,label=r'Mean Posterior probability $(p(C_k|$data)) + 2* Std. Dev.')]
##                               #,plt.Line2D([0],[0],color='k', marker='_', markersize=16 ,linestyle=' ', label='+/- 2* Standard deviation')]
##            ax.set_xticks(np.arange(Number_Of_Classes))
##            ax.set_xticklabels(reordered_names, rotation='vertical')
####            plt.subplots_adjust(bottom=0.25)
####            plt.gca().get_xticklabels()[k].set_fontweight('bold')
####            ax.yaxis.grid(True)
####            plt.ylim(0,1.05)
##            plt.xlabel(r'Classes $C_k$')
##            plt.ylabel(r'Posterior probability $p(C_k|$data)')
###            plt.legend(handles=Legend_elements)
##            if type(Testing_noise) == bool:
##                plt.savefig('Results/'+DataSet_Name+'/Noiseless/'+Model+'/'+Savename+'/snapshotfigurenew'+str(count)+'.pdf')
##            else:
##                plt.savefig('Results/'+DataSet_Name+'/Noise_'+str(Testing_noise)+'/'+Model+'/'+Savename+'/snapshotfigurenew'+str(count)+'.pdf')
##            plt.close()



            
##            fig2=sns.kdeplot(df2,multiple="layer")
##            plt.xlabel('P(C_k=Class| x )')
##            if type(Testing_noise) == bool:
##                plt.savefig('Results/'+DataSet_Name+'/Noiseless/'+Model+'/'+Savename+'/postdensdistribution'+'.pdf')
##            else:
##                plt.savefig('Results/'+DataSet_Name+'/Noise_'+str(Testing_noise)+'/'+Model+'/'+Savename+'/postdensdistribution'+str(count)+'.pdf')
##            plt.close()
##            print("Completed posterior plots")
##            del fig2

    
    print("Completed sampling")
    if k ==0:
        ProbabilitiesUpPL = pup
        ProbabilitiesLowPL = plow
        UQPL = uq
        ProbabilitiesPL = pmean
 

                   


    return ProbabilitiesUpPL,ProbabilitiesLowPL,UQPL,ProbabilitiesPL
