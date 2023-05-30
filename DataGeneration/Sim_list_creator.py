########################################################################
#This code was written by Ben Wilson with the supervison of Paul Ledger
#at the ZCCE, Swansea University
#Powered by NETGEN/NGSolve
#
#This code lists all of the object primitives that are available and
#compares it to the simulations which have already been run
########################################################################

























########################################################################
#Import
import os
import csv

#Function to edit floats to a nice format
def FtoS(value):
    if value==0:
        newvalue = "0"
    elif value==1:
        newvalue = "1"
    elif value==-1:
        newvalue = "-1"
    else:
        for i in range(100):
            if abs(value)<=1:
                if round(abs(value/10**(-i)),2)>=1:
                    power=-i
                    break
            else:
                if round(abs(value/10**(i)),2)<1:
                    power=i-1
                    break
        newvalue=value/(10**power)
        newvalue=str(round(newvalue,2))
        if newvalue[-1]=="0":
            newvalue=newvalue[:-1]
        if newvalue[-1]=="0":
            newvalue=newvalue[:-1]
        if newvalue[-1]==".":
            newvalue=newvalue[:-1]
        newvalue += "e"+str(power)

    return newvalue
#Main script


#Load the Current list of objects
with open('Material_Dictionary.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    Mat_dict = {}
    for row in reader:
        Mat_dict[row['Ref']]={'Mu':str(row['Mu']),'Sigma':str(row['Sigma'])}


#Create the list of all possible simulations to be run

#Load the list of objects (with materials listed)
with open('Object_List.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    Object_dict = {}
    for row in reader:
        if row['Materials']!=' ':
            Object_dict[row['Filepath']]={'Object':row['Object'],'Materials':row['Materials']}




#Create the list of simulations
Sweepnames = []
Sweepdict = {}
for filepath in Object_dict:
    OBJ = Object_dict[filepath]
    Mats = OBJ['Materials']
    
    #Create the material list
    if '[' in Mats:#Create the list for multiple materials
        Matlist = Mats.split(';[')
        for i,mat in enumerate(Matlist):
            Matlist[i] = mat.replace('[','').replace(']','')
    else:
        Matlist = Mats.split(';')

    #Create the sweeplist
    for Mat in Matlist:
        #Split the materials to be included
        mats = Mat.split(';')
        mus = ''
        fmus = []
        sigs = ''
        fsigs = []
        for mat in mats:
            Material = Mat_dict[mat]
            mus += Material['Mu']+','
            sigs += FtoS(float(Material['Sigma']))+','
            #store the values incase of dictionary addition
            fmus.append(float(Material['Mu']))
            fsigs.append(float(Material['Sigma']))
        mus = mus[:-1]
        sigs = sigs[:-1]
        
        filepath = filepath.replace('GeoFiles','Results')
        try:
            Sweeps = os.listdir(filepath)
            AddSweep = True
            for sweep in Sweeps:
                if '_mu_'+mus+'_sig_'+sigs in sweep:
                    AddSweep = False
        except:
            AddSweep = True
        if AddSweep == True:
            Sweepnames.append(filepath.replace('Results','GeoFiles'))
            Sweepdict[filepath.replace('Results','GeoFiles')+'_'+Mat]={'Mu':fmus,'Sig':fsigs}




#Write the file for the list of simulations
fieldnames = ['Filepath', 'Mu', 'Sig']
with open("Sim_List.csv", "w+") as f:
    w = csv.DictWriter(f, fieldnames)
    w.writeheader()
    for k in Sweepdict:
        w.writerow({field: Sweepdict[k].get(field) or k for field in fieldnames})


if len(Sweepnames)==1:
    print('There is {} simulation to run'.format(len(Sweepnames)))
else:
    print('There are {} simulations to run'.format(len(Sweepnames)))




















