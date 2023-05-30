########################################################################
#This code was written by Ben Wilson with the supervison of Paul Ledger
#at the ZCCE, Swansea University


# EDIT: 25/03/2022 - James Elgy:
# Removed for loop iterating over the invariants and replaced it with np.concatinate. This was to simplify the code and
# fix a bug concerning the indexing of NewData.
########################################################################


#User Inputs

#Dataset

#Dataset name
Name = 'Class_8/Class_8_1000'
#(string) Name of the dataset

#Frequencies to evaluate at
import numpy as np


Frequencies = np.array([119.25,178.875,238.5,298.125,357.75,477,596.25,715.5,954,1192.5,1431,1908,2385,2862,3816,4770,5724,7632,9540,12402,16218,20988,26712,34344,43884,57240,73458,95400])
Frequencies = Frequencies*6.28 #Convert to rad/s

# Loading in frequencies from John Davidson coin simulations:
#import pandas as pd
#filename = r'US Coin Dime MPTs.xlsx'
#external_data = pd.read_excel(filename)
#frequencies = external_data['Frequency']
#Frequencies = np.pi*2*frequencies


#Frequencies = np.logspace(np.log10(8*6.28*10**3),np.log10(13.8*6.28*10**3),20)





#Classes to include
#Coin Problem
#Classes = ['US_Coins']
#Classes = ['British_Coins']

#8 and 15 class problem
Classes = ['Realistic']
#(list) list of strings for the classes to be included in the training set
#this is hierarchic so includes all subclasses of a given class. Can also
#be 'all' to include all classes.


#Number of results

#Global number of results
#Coin Problem
#extend_results = 'global_obj'
#8 and 15 class problem
extend_results = 'global_class'
#(string) 'global_obj','global_class' or 'classwise' if 'objectwise' required use scaling file

#Number of secondary results if 'global_obj'
#Coin Problem
Num_Results = 50
#(int) Number of secondary results per object


#Classes if 'global_class' or 'classwise'
#8 class
class_split = [['Belt_Buckle','Shoe_shank'],['Coins','Keys'],['Bracelets','Watches'],'Earings','Pendants','Rings',['Hammers','Scissors','Screwdrivers'],['Guns','Knives','Knuckle_dusters']]
#class_split = ['US_Coins', 'British_Coins']
#15 class
#class_split = ['Belt_Buckle','Shoe_shank','Coins','Keys','Bracelets','Watches','Earings','Pendants','Rings','Hammers','Scissors','Screwdrivers','Guns','Knives','Knuckle_dusters']

#Name each of the classes this is done in order
#8 class
class_names = ['Clothing','Pocket Items','Wrist Items','Earrings','Pendants','Rings','Tools','Weapons']
#class_names = ['US Coins', 'British Coins']
#15 class
#class_names = ['Belt Buckles','Shoe shanks','Coins','Keys','Bracelets','Watches','Earrings','Pendants','Rings','Hammers','Scissors','Screwdrivers','Guns','Knives','Knuckle dusters']


#Number of results if 'global_class'
#8 and 15 class problem
Num_Results_class = 1000

#(int) Number of results per class results per class if 'classwise'
class_num_results = [150,100,100,100,200,100,100,300]
#(lists) list of strings or list of list of strings for the classes list of ints for the number of resutls


#Labeler False,'Classwise', 'Objectwise'
#Coin Problem
#Label_Data = 'Objectwise'
#8 and 15 class
Label_Data = 'Classwise'
#(boolean or string) create training labels for the dataset

#If Label_Data = 'Objectwise' you can create a dictionary of names for each object
#Coin Problem
Name_Objects = True
#(Boolean)

#Object Dictionary
#Coin Problem
#Object_Names_dictionary = {'Cent_Coin':r'Cent', 'Dime_Coin_Cladded_2':r'Dime', 'Nickel_Coin':r'Nickel', 'QuarterDollar_Coin_2':r'Quarter', 'HalfDollar_Coin_2':r'Half'}
Object_Names_dictionary ={'Two_Pound':r'£2', 'Ten_p_pre':r'10p', 'One_Pound':r'£1', 'Two_p_pre':r'2p', 'Twenty_p':r'20p', 'Five_p_pre':r'5p', 'Fifty_p':r'50p', 'One_p_pre':r'1p'}
#(dictionary) name the objects as you wish them to appear in the classificaiton

#Coin Problem
#Name_Order = ['Cent_Coin','Dime_Coin_Cladded_2','Nickel_Coin','QuarterDollar_Coin_2','HalfDollar_Coin_2']
Name_Order = ['One_p_pre','Two_p_pre','Five_p_pre','Ten_p_pre','Twenty_p','Fifty_p','One_Pound','Two_Pound']
#(list) list the order of the names as you wish them to appear in the classification


#Scaling

#How to scale
Scale_type = 'Global'
#(string) how to create secondry results: 'Global', 'File'

#Which file (This is not currently used)
Scale_File = 'Coin_DataSet.csv'
#(string) which file to use if Scale_type = 'File'

#Alpha scale
Alpha_scale = 0.84
#(float) percentage of original if Scale_type = 'Global' (this is a standard deviation)


#Sigma scale
Sigma_scale = 2.4
#(float) percentage of original if Scale_type = 'Global' (this is a standard deviation)



#From paper s_alpha = 8.4e-6 to obtain: Alpha_scale = (s_alpha/alpha)*100
#From paper s_sigma = 9.52e5 to obtain: Sigma_scale = (s_sigma/sigma)*100




Name += '_Al_'+str(Alpha_scale)+'_Sig_'+str(Sigma_scale)


########################################################################


#Main script


#Import
import os
import sys
import csv
import math
import subprocess
import matplotlib.pyplot as plt
from termcolor import colored

sys.path.insert(0,"Functions")
from Interpolater import *
from Scaler import *
from PreProcessors import *


#Print the file path of where the dataset is saved


print('The folder path to the dataset is:',Name)


#First check the naming works for the classwise naming

if type(Label_Data) != bool:
    if Label_Data == 'Classwise':
        Names_dic = {}
        class_dic = {}

        for i, name in enumerate(class_names):
            if type(class_split[i])==str:
                Names_dic[class_split[i]] = name
                class_dic[class_split[i]] = i
            else:
                for sub_class in class_split[i]:
                    Names_dic[sub_class] = name
                    class_dic[sub_class] = i




#Create the np.array containing the frequenices to be evalutated at
#Frequencies = np[10**5]
#Calculate the number of fields
#if lin_log == 'Lin':
#    Frequencies = np.linspace(10**Frequencies[0],10**Frequencies[1],Frequencies[2])
#else:
#    Frequencies = np.logspace(Frequencies[0],Frequencies[1],Frequencies[2])
Descriptors = 6 #for filepath, class, object, alpha, mur, sigma
Fields = 17*len(Frequencies) #6 Eigenvalues, 6 Principal, 4 Deviatoric, 1 Comutator
Num_Results -= 1


#Count the number of objects in each of the classes being created

if extend_results in ['classwise','global_class']:
    objects_per_class = np.zeros(len(class_split),dtype=int)
    class_dic = {}
    for i,cl in enumerate(class_split):
        if type(cl)==str:
            class_dic[cl] = i
        else:
            for sub_cl in cl:
                class_dic[sub_cl] = i
    for root, dirs, files in os.walk('Classes', topdown = True):
        for directory in dirs:
            if 'Class_' in directory and directory[6:] in class_dic.keys():
                for subroot,subdirs, subfiles in os.walk(root+'/'+directory,topdown = True):
                    for subdirectory in subdirs:
                        if 'OBJ_' in subdirectory:
                            Sweeps = os.listdir(subroot+'/'+subdirectory)
                            for Sweep in Sweeps:
                                if 'al_' in Sweep:
                                    objects_per_class[class_dic[directory[6:]]]+=1


    #work out how many scales per object should be done in each class
    if extend_results == 'classwise':
        scales_per_object = [class_num_results[i]/objects_per_class[i] for i in range(len(objects_per_class))]
    else:
        scales_per_object = [math.ceil(Num_Results_class/i) for i in objects_per_class]


    #create a dictionary with the number of scales to do
    class_dic_rev = {v: k for k, v in class_dic.items()}
    scales_per_class = {}
    for key in class_dic.keys():
        scales_per_class[key] = scales_per_object[class_dic[key]]






#Create the dataset
FirstInstance = True

if Scale_type == 'File':
    Scale_Dic = csv.DictReader(open("Coin_DataSet.csv",encoding="utf-8-sig"))
    Scale_List = []
    for OBJ in Scale_Dic:
        Scale_List.append(OBJ)

#Find the Classes
if type(Classes) == list:
    for root, dirs, files in os.walk('Classes', topdown = True):
        for directory in dirs:
            if 'Class_' in directory and directory[6:] in Classes:
                #Find the Objects in the classes
                for subroot, subdirs, subfiles in os.walk(root+'/'+directory,topdown = True):
                    for subdirectory in subdirs:
                        if 'OBJ_' in subdirectory:
                            Filepath = subroot+'/'+subdirectory
                            Class = subroot.replace('Classes/','').replace('Class_','')
                            Object = subdirectory.replace('OBJ_','')
                            Folders = os.listdir(Filepath)
                            Folders = [Folder for Folder in Folders if '.DS' not in Folder]
                            for Folder in Folders:
                                #Obtain the Alpha, Mur and Sigma values
                                if Folder[0] == '.':
                                    _,_,Al,_,Mur,_,Sig = Folder.split('_')
                                else:
                                    _,Al,_,Mur,_,Sig = Folder.split('_')
                                #Add to the instance descriptions
                                if FirstInstance == True:
                                    Descriptions = np.array([Filepath,Class,Object+'_Orig',Al,Mur,Sig])
                                else:
                                    Descriptions = np.vstack([Descriptions,np.array([Filepath,Class,Object+'_Orig',Al,Mur,Sig])])
                                Sweeps = os.listdir(Filepath+'/'+Folder)
                                Sweep = [Sweep for Sweep in Sweeps if '.DS' not in Sweep][0]

                                if Scale_type == 'File':
                                    Found = False
                                    for OBJ in Scale_List:
                                        if Object == OBJ['Object']:
                                            Found = True
                                            Num_Results = int(OBJ['Instances'])-1
                                            Alpha_scale = OBJ['Alpha_SD']
                                            Sigma_scale = OBJ['Sigma_SD']
                                            if '%' in Alpha_scale:
                                                Alpha_scale = float(Alpha_scale.replace('%',''))
                                            else:
                                                Alpha_scale = float(Alpha_scale)/float(Al)
                                            if '%' in Sigma_scale:
                                                Sigma_scale = float(Sigma_scale.replace('%',''))
                                            elif ',' in Sig:
                                                print(colored('      INPUT ERROR\n','red').center(os.get_terminal_size().columns))
                                                print('You cannot pass a value in S/m for the {} as it has multiple regions,'.format(Object).center(os.get_terminal_size().columns))
                                                sys.exit('it must be defined in terms of a %.'.center(os.get_terminal_size().columns))
                                            else:

                                                Sigma_scale = float(Sigma_scale)/float(Sig)

                                    if Found == False:
                                        print('Object attributes could not be found for {}'.format(Object))

                                if extend_results in ['classwise','global_class']:
                                    for key in scales_per_class.keys():
                                        if key in subroot:
                                            Num_Results = scales_per_class[key]-1

                                #Obtain and scale the primary result
                                OrigFreq, OrigEig, OrigTen, NewFreqs, NewEigs, NewTens, NewAls, NewSigs = Scale(Filepath,[Al],[Sig],Alpha_scale,Sigma_scale,int(Num_Results))

                                #Interpolate to the correct values
                                Tensors, Eigenvalues = LogInterp(OrigFreq,OrigEig,OrigTen,Frequencies,True)
                                ScaleTens = np.zeros([len(Frequencies),9,int(Num_Results)],dtype=complex)
                                ScaleEigs = np.zeros([len(Frequencies),3,int(Num_Results)],dtype=complex)
                                for i in range(int(Num_Results)):
                                    ScaleTens[:,:,i], ScaleEigs[:,:,i] = LogInterp(NewFreqs[:,i],NewEigs[:,:,i],NewTens[:,:,i],Frequencies,True)

                                #Create the features
                                Principal, Deviatoric, Z = FeatureCreation(Tensors,Eigenvalues)

                                #Add the features for the original sweep to the database
                                if FirstInstance == True:
                                    NewData = np.zeros([Fields])
                                    DataSet = np.zeros([Fields])
                                    Scaled_Tensors = Tensors
                                    NewData = np.concatenate((Eigenvalues[:,0].real, Eigenvalues[:,0].imag))
                                    NewData = np.concatenate((NewData, Eigenvalues[:, 1].real, Eigenvalues[:, 1].imag))
                                    NewData = np.concatenate((NewData, Eigenvalues[:, 2].real, Eigenvalues[:, 2].imag))
                                    NewData = np.concatenate((NewData, Principal[:, 0], Principal[:, 1], Principal[:, 2]))
                                    NewData = np.concatenate((NewData, Principal[:, 3], Principal[:, 4], Principal[:, 5]))
                                    NewData = np.concatenate((NewData, Deviatoric[:, 0], Deviatoric[:, 1], Deviatoric[:, 2], Deviatoric[:, 3]))
                                    NewData = np.concatenate((NewData, Z))

                                    DataSet[:] = NewData[:]
                                    FirstInstance = False
                                else:

                                    NewData = np.concatenate((Eigenvalues[:, 0].real, Eigenvalues[:, 0].imag))
                                    NewData = np.concatenate((NewData, Eigenvalues[:, 1].real, Eigenvalues[:, 1].imag))
                                    NewData = np.concatenate((NewData, Eigenvalues[:, 2].real, Eigenvalues[:, 2].imag))
                                    NewData = np.concatenate((NewData, Principal[:, 0], Principal[:, 1], Principal[:, 2]))
                                    NewData = np.concatenate((NewData, Principal[:, 3], Principal[:, 4], Principal[:, 5]))
                                    NewData = np.concatenate((NewData, Deviatoric[:, 0], Deviatoric[:, 1], Deviatoric[:, 2],Deviatoric[:, 3]))
                                    NewData = np.concatenate((NewData, Z))
                                    DataSet = np.vstack([DataSet,NewData])
                                    Scaled_Tensors = np.vstack([Scaled_Tensors,Tensors])

                                #Add the descriptions for the secondary sweeps
                                for j in range(int(Num_Results)):
                                    Descriptions = np.vstack([Descriptions,np.array([Filepath,Class,Object,NewAls[j],Mur,NewSigs[j]])])

                                #Add the features for the secondary sweeps to the database
                                for j in range(int(Num_Results)):
                                    Principal, Deviatoric, Z = FeatureCreation(ScaleTens[:,:,j],ScaleEigs[:,:,j])
                                    NewData = np.concatenate((Eigenvalues[:, 0].real, Eigenvalues[:, 0].imag))
                                    NewData = np.concatenate((NewData, Eigenvalues[:, 1].real, Eigenvalues[:, 1].imag))
                                    NewData = np.concatenate((NewData, Eigenvalues[:, 2].real, Eigenvalues[:, 2].imag))
                                    NewData = np.concatenate((NewData, Principal[:, 0], Principal[:, 1], Principal[:, 2]))
                                    NewData = np.concatenate((NewData, Principal[:, 3], Principal[:, 4], Principal[:, 5]))
                                    NewData = np.concatenate((NewData, Deviatoric[:, 0], Deviatoric[:, 1], Deviatoric[:, 2], Deviatoric[:, 3]))
                                    NewData = np.concatenate((NewData, Z))
                                    DataSet = np.vstack([DataSet,NewData])
                                    Scaled_Tensors = np.vstack([Scaled_Tensors,ScaleTens[:,:,j]])

else:
    for root, dirs, files in os.walk('Classes',topdown = True):
        for directory in dirs:
            if 'OBJ_' in directory:
                Filepath = root+'/'+directory
                Class = root.replace('Classes/','').replace('Class_','')
                Object = directory.replace('OBJ_','')
                Folders = os.listdir(Filepath)
                Folders = [Folder for Folder in Folders if '.DS' not in Folder]

                #Work out how many secondary sweep to produce if required
                if extend_results in ['classwise','global_class']:
                    for key in scales_per_class.keys():
                        if key in root:
                            Num_Results = scales_per_class[key]-1

                for Folder in Folders:
                    #Obtain the Alpha, Mur and Sigma values
                    if Folder[0] == '.':
                        _,_,Al,_,Mur,_,Sig = Folder.split('_')
                    else:
                        _,Al,_,Mur,_,Sig = Folder.split('_')
                    #Add to the instance descriptions
                    if FirstInstance == True:
                        Descriptions = np.array([Filepath,Class,Object+'_Orig',Al,Mur,Sig])
                    else:
                        Descriptions = np.vstack([Descriptions,np.array([Filepath,Class,Object+'_Orig',Al,Mur,Sig])])
                    Sweeps = os.listdir(Filepath+'/'+Folder)
                    Sweep = [Sweep for Sweep in Sweeps if '.DS' not in Sweep][0]

                    #Obtain and scale the primary result
                    OrigFreq, OrigEig, OrigTen, NewFreqs, NewEigs, NewTens, NewAls, NewSigs = Scale(Filepath,[Al],[Sig],Alpha_scale,Sigma_scale,Num_Results)

                    #Interpolate to the correct values
                    Tensors, Eigenvalues = LogInterp(OrigFreq,OrigEig,OrigTen,Frequencies,True)
                    ScaleTens = np.zeros([len(Frequencies),9,Num_Results],dtype=complex)
                    ScaleEigs = np.zeros([len(Frequencies),3,Num_Results],dtype=complex)
                    for i in range(Num_Results):
                        ScaleTens[:,:,i], ScaleEigs[:,:,i] = LogInterp(NewFreqs[:,i],NewEigs[:,:,i],NewTens[:,:,i],Frequencies,True)

                    #Create the features
                    Principal, Deviatoric, Z = FeatureCreation(Tensors,Eigenvalues)

                    #Add the features for the original sweep to the database
                    if FirstInstance == True:
                        NewData = np.zeros([Fields])
                        DataSet = np.zeros([Fields])
                        Scaled_Tensors = Tensors
                        for i in range(9):
                            if i<3:
                                NewData[len(Frequencies)*(i*2):len(Frequencies)*((i*2)+1)] = Eigenvalues[:,i].real
                                NewData[len(Frequencies)*((i*2)+1):len(Frequencies)*((i*2)+2)] = Eigenvalues[:,i].imag
                            elif i<6:
                                NewData[len(Frequencies)*(i*2):len(Frequencies)*((i*2)+1)] = Principal[:,i-3]
                                NewData[len(Frequencies)*((i*2)+1):len(Frequencies)*((i*2)+2)] = Principal[:,i-2]
                            elif i<8:
                                NewData[len(Frequencies)*(i*2):len(Frequencies)*((i*2)+1)] = Deviatoric[:,i-6]
                                NewData[len(Frequencies)*((i*2)+1):len(Frequencies)*((i*2)+2)] = Deviatoric[:,i-5]
                            else:
                                NewData[len(Frequencies)*(i*2):len(Frequencies)*((i*2)+1)] = Z
                        DataSet[:] = NewData[:]
                        FirstInstance = False
                    else:
                        for i in range(9):
                            if i<3:
                                NewData[len(Frequencies)*(i*2):len(Frequencies)*((i*2)+1)] = Eigenvalues[:,i].real
                                NewData[len(Frequencies)*((i*2)+1):len(Frequencies)*((i*2)+2)] = Eigenvalues[:,i].imag
                            elif i<6:
                                NewData[len(Frequencies)*(i*2):len(Frequencies)*((i*2)+1)] = Principal[:,i-3]
                                NewData[len(Frequencies)*((i*2)+1):len(Frequencies)*((i*2)+2)] = Principal[:,i-2]
                            elif i<8:
                                NewData[len(Frequencies)*(i*2):len(Frequencies)*((i*2)+1)] = Deviatoric[:,i-6]
                                NewData[len(Frequencies)*((i*2)+1):len(Frequencies)*((i*2)+2)] = Deviatoric[:,i-5]
                            else:
                                NewData[len(Frequencies)*(i*2):len(Frequencies)*((i*2)+1)] = Z
                        DataSet = np.vstack([DataSet,NewData])
                        Scaled_Tensors = np.vstack([Scaled_Tensors,Tensors])


                    #Add the descriptions for the secondary sweeps
                    for j in range(Num_Results):
                        Descriptions = np.vstack([Descriptions,np.array([Filepath,Class,Object,NewAls[j],Mur,NewSigs[j]])])


                    #Add the features for the secondary sweeps to the database
                    for j in range(Num_Results):
                        Principal, Deviatoric, Z = FeatureCreation(ScaleTens[:,:,j],ScaleEigs[:,:,j])
                        for i in range(9):
                            if i<3:
                                NewData[len(Frequencies)*(i*2):len(Frequencies)*((i*2)+1)] = ScaleEigs[:,i,j].real
                                NewData[len(Frequencies)*((i*2)+1):len(Frequencies)*((i*2)+2)] = ScaleEigs[:,i,j].imag
                            elif i<6:
                                NewData[len(Frequencies)*(i*2):len(Frequencies)*((i*2)+1)] = Principal[:,i-3]
                                NewData[len(Frequencies)*((i*2)+1):len(Frequencies)*((i*2)+2)] = Principal[:,i-2]
                            elif i<8:
                                NewData[len(Frequencies)*(i*2):len(Frequencies)*((i*2)+1)] = Deviatoric[:,i-6]
                                NewData[len(Frequencies)*((i*2)+1):len(Frequencies)*((i*2)+2)] = Deviatoric[:,i-5]
                            else:
                                NewData[len(Frequencies)*(i*2):len(Frequencies)*((i*2)+1)] = Z
                        DataSet = np.vstack([DataSet,NewData])
                        Scaled_Tensors = np.vstack([Scaled_Tensors,ScaleTens[:,:,j]])


#Save the data
#Make the folder
try:
    os.makedirs('DataSets/'+Name)
except:
    pass

#Make the files for loading
with open('DataSets/'+Name+'/Descriptions.npy', 'wb') as f:
    np.save(f, Descriptions)
with open('DataSets/'+Name+'/DataSet.npy', 'wb') as f:
    np.save(f, DataSet)
Scaled_Tensors = Scaled_Tensors.reshape(int(np.shape(Scaled_Tensors)[0]/len(Frequencies)),len(Frequencies),9)
with open('DataSets/'+Name+'/Tensors.npy', 'wb') as f:
    np.save(f, Scaled_Tensors)

#Create a file with the frequencies sampled at
np.savetxt('DataSets/'+Name+'/Frequencies.csv',Frequencies,delimiter = ',')

#Make a readable file
np.savetxt('DataSets/'+Name+'/Descriptions.csv',Descriptions,delimiter = ' ', fmt="%s")
np.savetxt('DataSets/'+Name+'/DataSet.csv',DataSet,delimiter = ',')








#Create an overview of the data
OBJ_Overview = set()
for i in range(np.shape(Descriptions)[0]):
    OBJ_Overview.add(Descriptions[i,2].replace('_Orig',''))

#Create the array to store the data
#Object name, number of objects, alpha sample mean, alpha sample SD, sigma sample mean, sigma sample SD
Overview = []
for OBJ in OBJ_Overview:
    Overview.append([OBJ,0,0,0,0,0])
for i in range(np.shape(Descriptions)[0]):
    for j,OBJ in enumerate(OBJ_Overview):
        if OBJ == Descriptions[i,2].replace('_Orig',''):
            Overview[j][1] += 1
            Overview[j][2] += float(Descriptions[i,3])
            if len(Descriptions[i,5].split(',')) == 1:
                Overview[j][4] += float(Descriptions[i,5])
            else:
                if Overview[j][1] == 1:
                    Overview[j][4] = list(map(float,Descriptions[i,5].split(',')))
                else:
                    Overview[j][4] = [Overview[j][4][k]+list(map(float,Descriptions[i,5].split(',')))[k] for k in range(len(Overview[j][4]))]

#Create the means
for i in range(len(OBJ_Overview)):
    Overview[i][2] = Overview[i][2]/Overview[i][1]
    if type(Overview[i][4]) == float:
        Overview[i][4] = Overview[i][4]/Overview[i][1]
    else:
        Overview[i][4] = [Overview[i][4][j]/Overview[i][1] for j in range(len(Overview[i][4]))]

#Calculate the SDs
for i in range(np.shape(Descriptions)[0]):
    for j,OBJ in enumerate(OBJ_Overview):
        if OBJ in Descriptions[i,2]:
            Overview[j][3] += (float(Descriptions[i][3])-Overview[j][2])**2
            if len(Descriptions[i,5].split(',')) == 1:
                Overview[j][5] += (float(Descriptions[i][5])-Overview[j][4])**2
            else:
                if type(Overview[j][5]) == int:
                    Overview[j][5] = [(list(map(float,Descriptions[i][5].split(',')))[k]-Overview[j][4][k])**2 for k in range(len(Overview[j][4]))]
                else:
                    Overview[j][5] = [Overview[j][5][k]+(list(map(float,Descriptions[i][5].split(',')))[k]-Overview[j][4][k])**2 for k in range(len(Overview[j][4]))]

#Create the mean of the squares
for i in range(len(OBJ_Overview)):
    if Overview[i][1] == 1:
        Overview[i][3] = 0
        if type(Overview[i][4]) == float:
            Overview[i][5] = 0
        else:
            Overview[i][5] = [0 for j in range(len(Overview[i][5]))]
    else:
        Overview[i][3] = (Overview[i][3]/(Overview[i][1]-1))**(1/2)
        if type(Overview[i][4]) == float:
            Overview[i][5] = (Overview[i][5]/(Overview[i][1]-1))**(1/2)
        else:
            Overview[i][5] = [(Overview[i][5][j]/(Overview[i][1]-1))**(1/2) for j in range(len(Overview[i][5]))]

#Create the title and save
Overview_with_title = [['Object','Quantity','Alpha mean','Alpha SD','Sigma mean','Sigma SD']]
for line in Overview:
    Overview_with_title.append(line)
with open('DataSets/'+Name+'/Data_Overview.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(Overview_with_title)




if type(Label_Data) != bool:
    if Label_Data == 'Classwise':
        Labels = []
        Names = []
        for Instance in Descriptions:
            for key in class_dic.keys():
                if key in Instance[1]:
                    Labels.append(class_dic[key])
                    Names.append(Names_dic[key])

    elif Label_Data == 'Objectwise':
        Objects = {}
        if Name_Objects == True:
            for i,name in enumerate(Name_Order):
                Objects.update({name:i})
        else:
            p = 0
            for Instance in Descriptions:
                if '_Orig' in Instance[2]:
                    Objects.update({Instance[2].replace('_Orig',''):p})
                    p += 1

        Labels = [Objects[Inst[2]] if '_Orig' not in Inst[2] else Objects[Inst[2].replace('_Orig','')] for Inst in Descriptions]

        if Name_Objects == True:
            Names = [Object_Names_dictionary[Inst[2]] if '_Orig' not in Inst[2] else Object_Names_dictionary[Inst[2].replace('_Orig','')] for Inst in Descriptions]
        else:
            Names = Descriptions[:,2]


    np.savetxt('DataSets/'+Name+'/Labels.csv',Labels,delimiter = ',', fmt="%s")
    np.savetxt('DataSets/'+Name+'/names.csv',Names,delimiter = ',', fmt="%s")
