########################################################################
#This code was written by Ben Wilson with the supervison of Paul Ledger
#at the ZCCE, Swansea University
#Powered by NETGEN/NGSolve
########################################################################






########################################################################
#Import
import csv
from Sim_Run import *

#Main script

#Load the Current list of objects
with open('Sim_List.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    Sim_dict = {}
    for row in reader:
        Sim_dict[row['Filepath']]={'Mu':str(row['Mu']),'Sigma':str(row['Sig'])}



for i,sim in enumerate(Sim_dict):
    Good_Run = Sim_Runner(sim,Sim_dict[sim]['Mu'],Sim_dict[sim]['Sigma'])
















