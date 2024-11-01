import joblib
import numpy as np
import pandas as pd # Pandas is used for loading in the .xlsx files John provided.
from matplotlib import pyplot as plt
import seaborn as sns # Seaborn is a plotting library closely integrated with pandas. I am using it for plotting the
# comparisons plots.

"""
James Elgy - 2022:
Data loader for external MPT eigenvalues from excel files.
Currently loads in data from disk and reformats it to match the Input_Array format used in Trainer.py. The resultant
X_input array is then saved to disk.

The script requires that filenames be specified and the appropriate label for each object is provided. In addition the
features of interest must be specified and the frequencies used should match those of Creator.py.
"""

"""
Paul Ledger - May 2024 edit:
Converted to function so that it interfaces with a jupyter notebook. Allows for reduced confusion/errors when setting
the features etc.

Converted to input dataframe rather than file - useful for testing classifier with unseen but simulated data
"""

def external_file_loader_dataframe(Loader_Settings):

    # Obtain the problem parameters from the Loader_Settings dictionary

    dataframes = Loader_Settings["dataframes"]

    classes = Loader_Settings["classes"]

    Features = Loader_Settings["Features"]

    dataset_path = Loader_Settings["dataset_path"]

    result_path = Loader_Settings["result_path"]

    plot_comparison = Loader_Settings["plot_comparison"]




    count = 0
    # For each df and label, calculate eigenvalues and invariants and write to disk.
    for df_class_pair in zip(dataframes, classes):
        external_data = df_class_pair[0]
        label = df_class_pair[1]
        #external_data = pd.read_excel(filename)
        frequencies = external_data['Frequency']
        omega = frequencies
        Frequencies = frequencies
        #omega = np.pi*2*frequencies # no conversion needed as simulated data in rad/s
        #Frequencies = omega
        eig1 = external_data['Eigenvalue 1']
        eig2 = external_data['Eigenvalue 2']
        eig3 = external_data['Eigenvalue 3']
        angles=external_data['Angles']
    
        # Converting from str to complex array.
        #for eig in external_data['Eigenvalue 1']:
        #    string = str(eig)
        #    string = string.replace(' ', '')
        #    string = string.replace('i','j')
        #    print(string)
        #    eig1 += [complex(string)]
        #for eig in external_data['Eigenvalue 2']:
        #    string = str(eig)
        #    string = string.replace(' ', '')
        #    string = string.replace('i','j')
        #    print(string)
        #    eig2 += [complex(string)]
        #for eig in external_data['Eigenvalue 3']:
        #    string = str(eig)
        #    string = string.replace(' ', '')
        #    string = string.replace('i','j')
        #    print(string)
        #    eig3 += [complex(string)]
        #print("Angles" in external_data.columns) 
        #if ("Angles" in external_data.columns) == False:
        #    print("angles not in data frame")
        #    # Angles are not present in dataframe
        #    for eig in external_data['Eigenvalue 3']:
        #        angles +=[0]
        #else:
        #    for ang in external_data['Angles']:
        #        string = str(ang)
        #        string = string.replace(' ', '')
        #        string = string.replace('i','j')
        #        print(string)
        #        angles += [float(string)]

        eig1 = np.asarray(eig1)
        eig3 = np.asarray(eig2)
        eig2 = np.asarray(eig3)
        angles = np.asarray(angles)

        # Seperating real and imag components for the calculation of the invariants.
        # Ben has stored eigenvalues as real and imaginary components (e.g. eig1 = R + jI) but we should think of these as
        # seperate eigenvalues rather than a single complex number. I.e an eigenvalue of the real tensor and the eigenvalue
        # of the imaginary tensor.

        # Spliting eigenvalues into seperate variables to make the calculation more explicit.
        eig1_real = eig1.real
        eig1_imag = eig1.imag
        eig2_real = eig2.real
        eig2_imag = eig2.imag
        eig3_real = eig3.real
        eig3_imag = eig3.imag

        pri1_real = eig1_real + eig2_real + eig3_real
        pri2_real = eig1_real*eig2_real + eig1_real*eig3_real + eig2_real*eig3_real
        pri3_real = eig1_real*eig2_real*eig3_real
        pri1_imag = eig1_imag + eig2_imag + eig3_imag
        pri2_imag = eig1_imag*eig2_imag + eig1_imag*eig3_imag + eig2_imag*eig3_imag
        pri3_imag = eig1_imag*eig2_imag*eig3_imag

        dev2_real = 0.5 * ( (eig1_real - pri1_real/3)**2 + (eig2_real - pri1_real/3)**2 + (eig3_real - pri1_real/3)**2)
        dev3_real = (eig1_real - pri1_real/3)*(eig2_real-pri1_real/3)*(eig3_real-pri1_real)/3
        dev2_imag = 0.5 * ( (eig1_imag - pri1_imag/3)**2 + (eig2_imag - pri1_imag/3)**2 + (eig3_imag - pri1_imag/3)**2)
        dev3_imag = (eig1_imag - pri1_imag/3)*(eig2_imag-pri1_imag/3)*(eig3_imag-pri1_imag)/3
        com = np.zeros(dev3_real.shape, dtype=complex)

        # Following the same style as MPT_Calculator, the invariants are stored as complex floats.
        pri1 = pri1_real + 1j*pri1_imag
        pri2 = pri2_real + 1j*pri2_imag
        pri3 = pri3_real + 1j*pri3_imag
        dev2 = dev2_real + 1j*dev2_imag
        dev3 = dev3_real + 1j*dev3_imag

        # constructing input array as a real 1x(N_freq x 2 x N_features) array.
        input_array = eig1.real
        input_array = np.concatenate((input_array, eig1.imag))
        input_array = np.concatenate((input_array, eig2.real, eig2.imag))
        input_array = np.concatenate((input_array, eig3.real, eig3.imag))

        input_array = np.concatenate((input_array, pri1.real, pri1.imag))
        input_array = np.concatenate((input_array, pri2.real, pri2.imag))
        input_array = np.concatenate((input_array, pri3.real, pri3.imag))

        input_array = np.concatenate((input_array, dev2.real, dev2.imag))
        input_array = np.concatenate((input_array, dev3.real, dev3.imag))

        input_array = np.concatenate((input_array, com.real))
        input_array = np.concatenate((input_array, angles.real))
        
        input_array = input_array[None,:]
        print(np.shape(input_array))
        # np.save(dataset_path + '/Input_Array', input_array)

        # Building Feature Data:
        #Feature_Dic = {'Eig1': 0, 'Eig2': 1, 'Eig3': 2, 'Pri1': 3, 'Pri2': 4, 'Pri3': 5, 'Dev2': 6, 'Dev3': 7, 'Com': 8}
        #Create the desired features
        Feature_Size = 0
        for Feat in Features:
            if Feat == 'Com' or Feat == 'AngleRtildeI':
                Feature_Size += 1 # These features are real valued only
            else:
                Feature_Size += 2 # Other features are complex and so we need to consider real and imaginary parts
        Feature_Data = np.zeros([np.shape(input_array)[0],Feature_Size*len(Frequencies)])
        #Create a dictionary for Feature selection
        Feature_Dic = {'Eig1' : 0, 'Eig2' : 1, 'Eig3' : 2, 'Pri1' : 3, 'Pri2' : 4, 'Pri3' : 5, 'Dev2' : 6, 'Dev3' : 7, 'Com' : 8, 'AngleRtildeI': 9}

        #Create the Features and Labels to be used
        count2=0
        # Code updated since we are not always dealing with real and imaginary parts for
        # Feature_Dic[Feature]=8,9 then we have only a real part.
        for i,Feature in enumerate(Features):
            #print(Feature,Feature_Dic[Feature],len(Frequencies)*2*Feature_Dic[Feature],len(Frequencies)*2*(Feature_Dic[Feature]+1))
            if Feature_Dic[Feature] < 8:
        #    Feature_Data[:,len(Frequencies)*2*i:len(Frequencies)*2*(i+1)] = Data[:,len(Frequencies)*2*Feature_Dic[Feature]:len(Frequencies)*2*(Feature_Dic[Feature]+1)]
                Feature_Data[:,count2:count2+len(Frequencies)*2] = input_array[:,len(Frequencies)*2*Feature_Dic[Feature]:len(Frequencies)*2*(Feature_Dic[Feature]+1)]
                count2+=len(Frequencies)*2
            elif Feature_Dic[Feature]==8:
                Feature_Data[:,count2:count2+len(Frequencies)]=input_array[:,len(Frequencies)*2*Feature_Dic[Feature]:len(Frequencies)*2*(Feature_Dic[Feature])+len(Frequencies)]
                count2+=len(Frequencies)
            elif Feature_Dic[Feature]==9:
                Feature_Data[:,count2:count2+len(Frequencies)]=input_array[:,len(Frequencies)*2*(Feature_Dic[Feature]-1)+len(Frequencies):len(Frequencies)*2*(Feature_Dic[Feature]-1)+2*len(Frequencies)]
                count2+=len(Frequencies)
        #print(np.shape(Feature_Data))
        # Feature_Data holds only the features of interest.
#        Feature_Data = np.zeros([np.shape(input_array)[0], (2*len(Features)) * len(omega)])
#        for i, Feature in enumerate(Features):
#            Feature_Data[:, len(omega) * 2 * i:len(omega) * 2 * (i + 1)] = input_array[:,
#                                                                                       len(omega) * 2 * Feature_Dic[
#                                                                                           Feature]:len(omega) * 2 * (
#                                                                                                   Feature_Dic[Feature] + 1)]

        # labels = np.genfromtxt(dataset_path + '/Labels.csv')
        # names = np.genfromtxt(dataset_path + '/names.csv', dtype=str)
        # names_flag = [1 if name=='Dime' else 0 for name in names]
        # label_val = [labels[ind] if name =='Dime' else 0 for ind, name in enumerate(names)]
        # label = max(label_val)

        X_input = np.zeros((1,Feature_Data.shape[1]+2))
        X_input[0,0] = count
        X_input[0,1:-1] = Feature_Data
        X_input[0,-1] = label
        if count > 0:
            full_data = np.vstack((full_data, X_input))
        else:
            full_data = X_input
        count += 1

    # Saving out array of test data to disk. This will be loaded back into memory when Trainer.py is run using
    # Load_External_Data.
    np.savetxt(dataset_path + '/X_Input.csv', full_data)
    print(np.shape(X_input),np.shape(full_data))
    ### FOR COMPARISON ###
    if (plot_comparison is True) and (len(filenames) == 1):
        try:
            internal_data = np.genfromtxt(results_path + '/Input_Array.csv', delimiter=',')
        except:
            raise NameError('No internal input array found. Run Trainer.py with Full_Save.')
        feature1_real = internal_data[:,1:len(omega)+1]
        feature1_imag = internal_data[:,len(omega)+1:2*len(omega)+1]

        feature2_real = internal_data[:,2*len(omega)+1:3*len(omega)+1]
        feature2_imag = internal_data[:,3*len(omega)+1:4*len(omega)+1]

        feature3_real = internal_data[:,4*len(omega)+1:5*len(omega)+1]
        feature3_imag = internal_data[:,5*len(omega)+1:6*len(omega)+1]

        object_class = internal_data[:,-1][:,None] @ np.ones((1,71))
        object_omega = np.ones((internal_data.shape[0],1)) @ omega[None,:]

        internal_data_dict = {'pri1_real': np.ravel(feature1_real), 'pri1_imag': np.ravel(feature1_imag),
                              'pri2_real': np.ravel(feature2_real), 'pri2_imag': np.ravel(feature2_imag),
                              'pri3_real': np.ravel(feature3_real), 'pri3_imag': np.ravel(feature3_imag),
                              'omega': np.ravel(object_omega), 'class': np.ravel(object_class)}

        internal_dataframe = pd.DataFrame(internal_data_dict)

        plt.figure()
        internal_lineplot = sns.lineplot(data=internal_dataframe[internal_dataframe['class'] == 1], x='omega', y='pri2_real')
        internal_lineplot.set(xscale='log')
        plt.plot(omega, pri2.real, label='external_data')
        # plt.plot(omega, X_input[0,1:72])
        plt.legend()
