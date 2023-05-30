########################################################################
#This code was written by Ben Wilson with the supervison of Paul Ledger
#at the ZCCE, Swansea University
#Powered by NETGEN/NGSolve
#
#This code lists all of the object primitives that are available and 
#updates the Object_Dictionary accordingly
########################################################################







########################################################################
#Import
import os
import csv
#Main script

#Load the Current list of objects
with open('Object_List.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    Object_dict = {}
    for row in reader:
        Object_dict[row['Filepath']]={'Object':row['Object'],'Materials':row['Materials']}



#Cycle through the different objects and add them if necessary
for root, dirs, files in os.walk('GeoFiles',topdown = True):
    for directory in dirs:
        if 'OBJ_' in directory:
            Filepath = root+'/'+directory
            Object = directory.replace('OBJ_','')
            if Filepath not in Object_dict.keys():
                Object_dict[Filepath]={'Object':Object,'Materials':' '}


#Rewrite the file
fieldnames = ['Filepath', 'Object', 'Materials']
with open("Object_List.csv", "w+") as f:
    w = csv.DictWriter(f, fieldnames)
    w.writeheader()
    for k in Object_dict:
        w.writerow({field: Object_dict[k].get(field) or k for field in fieldnames})


