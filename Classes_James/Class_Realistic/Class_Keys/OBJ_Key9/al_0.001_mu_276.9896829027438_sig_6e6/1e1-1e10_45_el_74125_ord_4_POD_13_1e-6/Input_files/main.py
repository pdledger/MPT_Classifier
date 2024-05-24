########################################################################
#This code was written by Ben Wilson with the supervison of Paul Ledger
#at the ZCCE, Swansea University
#Powered by NETGEN/NGSolve
########################################################################

# Importing
import sys
import time
from memory_profiler import memory_usage
import numpy as np
import os
#import matplotlib
#matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt

sys.path.insert(0, "Functions")
sys.path.insert(0, "Settings")
from MeshCreation import *
from Settings import *
from SingleSolve import SingleFrequency
from FullSolvers import *
from PODSolvers import *
from ResultsFunctions import *
from Checkvalid import *
from JamesFunctions import *

# from ngsolve import ngsglobals
# ngsglobals.msg_level = 0


def main(hp=(), curve_degree=5, return_settings=False, start_stop=(), alpha='', geofile='default', frequency_array='default', from_py_file=False):
    if len(hp) == 2:
        OVERWRITE_HP = True
    else:
        OVERWRITE_HP = False

    if return_settings == True:
        SETTINGS_ONLY = True
    else:
        SETTINGS_ONLY = False

    if len(start_stop) == 3:
        OVERWRITE_START_STOP = True
    else:
        OVERWRITE_START_STOP = False

    if type(alpha) == int or type(alpha) == float:
        OVERWRITE_ALPHA = True
    else:
        OVERWRITE_ALPHA = False

    #User Inputs

    #Geometry
    #(string) Name of the .geo file to be used in the frequency sweep i.e.
    # "sphere.geo"
    if geofile != 'default':
        Geometry = geofile
    else:
        Geometry = 'sphere.geo'
    print(Geometry)
    #Scaling to be used in the sweep in meters
    # alpha = 1e-2
    if OVERWRITE_ALPHA == False:
        alpha = 1e-3
    #(float) scaling to be applied to the .geo file i.e. if you have defined
    #a sphere of unit radius in a .geo file   alpha = 0.01   would simulate a
    #sphere with a radius of 0.01m ( or 1cm)


    #About the mesh
    #How fine should the mesh be
    MeshSize = 2
    if OVERWRITE_HP:
        MeshSize = hp[0]
    #(int 1-5) this defines how fine the mesh should be for regions that do
    #not have maxh values defined for them in the .geo file (1=verycoarse,
    #5=veryfine)

    #The order of the elements in the mesh
    Order = 2
    if OVERWRITE_HP:
        Order = hp[1]
    #(int) this defines the order of each of the elements in the mesh


    #About the Frequency sweep (frequencies are in radians per second)
    #Minimum frequency (Powers of 10 i.e Start = 2 => 10**2)
    Start = 1
    #(float)
    #Maximum frequency (Powers of 10 i.e Start = 8 => 10**8)
    Finish = 10

    if OVERWRITE_START_STOP:
        Start = start_stop[0]
        Finish = start_stop[1]
    #(float)
    #Number of points in the freqeuncy sweep
    Points = 45

    if OVERWRITE_START_STOP:
        Points = start_stop[2]
    #(int) the number of logarithmically spaced points in the sweep

    #I only require a single frequency
    Single = False
    #(boolean) True if single frequency is required
    Omega = 23976
    #(float) the frequency to be solved if Single = True


    #POD
    #I want to use POD in the frequency sweep
    Pod = True
    #(boolean) True if POD is to be used, the number of snapshots can be
    #edited in in the Settings.py file


    #MultiProcessing
    MultiProcessing = False
    #(boolean) #I have multiple cores at my disposal and have enough spare RAM
    # to run the frequency sweep in parrallel (Edit the number of cores to be
    #used in the Settings.py file)
    if SETTINGS_ONLY:
        return (Geometry, alpha, MeshSize, Order, Start, Finish, Points, Single, Omega, Pod, MultiProcessing)

    ########################################################################
    #Main script

    #Load the default settings
    CPUs,BigProblem,PODPoints,PODTol,OldMesh = DefaultSettings()

    if OldMesh == False:
        #Create the mesh
        Meshmaker(Geometry,MeshSize)
    else:
        #Check whether to add the material information to the .vol file
        try:
            Materials,mur,sig,inorout = VolMatUpdater(Geometry,OldMesh)
            ngmesh = ngmeshing.Mesh(dim=3)
            ngmesh.Load("VolFiles/"+Geometry[:-4]+".vol")
            mesh = Mesh("VolFiles/"+Geometry[:-4]+".vol")
            mu_coef = [ mur[mat] for mat in mesh.GetMaterials() ]
        except:
            #Force update to the .vol file
            OldMesh = False

    #Update the .vol file and create the material dictionaries
    Materials,mur,sig,inorout = VolMatUpdater(Geometry,OldMesh)

    #create the arraFalsey of points to be used in the sweep
    Array = np.logspace(Start,Finish,Points)
    if frequency_array != 'default':
        Array = frequency_array
    PlotPod, PODErrorBars, EddyCurrentTest, vtk_output, Refine = AdditionalOutputs()
    SavePOD = False
    if PODErrorBars!=True:
        ErrorTensors=False
    else:
        ErrorTensors=True
    PODArray = np.logspace(Start,Finish,PODPoints)

    #Create the folders which will be used to save everything
    sweepname = FolderMaker(Geometry, Single, Array, Omega, Pod, PlotPod, PODArray, PODTol, alpha, Order, MeshSize, mur, sig, ErrorTensors, vtk_output)


    #Run the sweep

    #Check the validity of the eddy-current model for the object
    if EddyCurrentTest == True:
        EddyCurrentTest = Checkvalid(Geometry,Order,alpha,inorout,mur,sig)

    if Single==True:
        if MultiProcessing!=True:
            CPUs = 1
        MPT, EigenValues, N0, elements, ndofs = SingleFrequency(Geometry,Order,alpha,inorout,mur,sig,Omega,CPUs,vtk_output,Refine, curve=curve_degree)
        TensorArray = MPT
    else:
        if Pod==True:
            if MultiProcessing==True:
                if PlotPod==True:
                    if PODErrorBars==True:
                        TensorArray, EigenValues, N0, PODTensors, PODEigenValues, elements, ErrorTensors = PODSweepMulti(Geometry,Order,alpha,inorout,mur,sig,Array,PODArray,PODTol,PlotPod,CPUs,sweepname,SavePOD,PODErrorBars,BigProblem)
                    else:
                        TensorArray, EigenValues, N0, PODTensors, PODEigenValues, elements = PODSweepMulti(Geometry,Order,alpha,inorout,mur,sig,Array,PODArray,PODTol,PlotPod,CPUs,sweepname,SavePOD,PODErrorBars,BigProblem)
                else:
                    if PODErrorBars==True:
                        TensorArray, EigenValues, N0, elements, ErrorTensors = PODSweepMulti(Geometry,Order,alpha,inorout,mur,sig,Array,PODArray,PODTol,PlotPod,CPUs,sweepname,SavePOD,PODErrorBars,BigProblem)
                    else:
                        TensorArray, EigenValues, N0, elements = PODSweepMulti(Geometry,Order,alpha,inorout,mur,sig,Array,PODArray,PODTol,PlotPod,CPUs,sweepname,SavePOD,PODErrorBars,BigProblem)
            else:
                if PlotPod==True:
                    if PODErrorBars==True:
                        TensorArray, EigenValues, N0, PODTensors, PODEigenValues, elements, ErrorTensors = PODSweep(Geometry,Order,alpha,inorout,mur,sig,Array,PODArray,PODTol,PlotPod,sweepname,SavePOD,PODErrorBars,BigProblem)
                    else:
                        TensorArray, EigenValues, N0, PODTensors, PODEigenValues, elements = PODSweep(Geometry,Order,alpha,inorout,mur,sig,Array,PODArray,PODTol,PlotPod,sweepname,SavePOD,PODErrorBars,BigProblem)
                else:
                    if PODErrorBars==True:
                        TensorArray, EigenValues, N0, elements, ErrorTensors = PODSweep(Geometry,Order,alpha,inorout,mur,sig,Array,PODArray,PODTol,PlotPod,sweepname,SavePOD,PODErrorBars,BigProblem)
                    else:
                        TensorArray, EigenValues, N0, elements = PODSweep(Geometry,Order,alpha,inorout,mur,sig,Array,PODArray,PODTol,PlotPod,sweepname,SavePOD,PODErrorBars,BigProblem)
            ndofs = -1

        else:
            if MultiProcessing==True:
                TensorArray, EigenValues, N0, elements, ndofs = FullSweepMulti(Geometry,Order,alpha,inorout,mur,sig,Array,CPUs,BigProblem)
            else:
                TensorArray, EigenValues, N0, elements, ndofs = FullSweep(Geometry,Order,alpha,inorout,mur,sig,Array,BigProblem)


    #Plotting and saving
    if Single==True:
        SingleSave(Geometry, Omega, MPT, EigenValues, N0, elements, alpha, Order, MeshSize, mur, sig, EddyCurrentTest)
    elif PlotPod==True:
        if Pod==True:
            PODSave(Geometry, Array, TensorArray, EigenValues, N0, PODTensors, PODEigenValues, PODArray, PODTol, elements, alpha, Order, MeshSize, mur, sig, ErrorTensors,EddyCurrentTest)
        else:
            FullSave(Geometry, Array, TensorArray, EigenValues, N0, Pod, PODArray, PODTol, elements, alpha, Order, MeshSize, mur, sig, ErrorTensors,EddyCurrentTest)
    else:
        FullSave(Geometry, Array, TensorArray, EigenValues, N0, Pod, PODArray, PODTol, elements, alpha, Order, MeshSize, mur, sig, ErrorTensors,EddyCurrentTest)

    return TensorArray, EigenValues, N0, elements, Array, ndofs, EddyCurrentTest


def overwrite_tag_value(filename, tag, value):
    """
    Function to read through and replace value for tag. i.e replace mur=1 with mur=2.
    :param filename: file name for .geo file in question
    :param tag: tag to search for
    :param value: new value
    """
    snippet = '-' + tag + '='
    f = open(filename, 'r')
    flines = f.readlines()
    for index, line in enumerate(flines):
        place = line.find(snippet)
        if place > 0:
            replacement_characters = str(value)
            tag_start = place
            tag_stop = line[place + 1:].find('-')
            if tag_stop == -1:  # tag is at end of line
                tag_stop = line[place + 1:].find('\n') + 1
            replacement_tag = snippet + replacement_characters + ' '
            replacement_line = line.replace(line[tag_start:tag_start + tag_stop], replacement_tag)

            flines[index] = replacement_line
    with open(filename, 'w') as f:
        f.writelines(flines)


if __name__ == '__main__':

    m = 200
    s = 100/2

    mur = np.random.normal(m,s,10)

    for inst in mur:
        overwrite_tag_value('GeoFiles/remeshed_matt_key_9_steel.geo', 'mur', inst)
        print(f'solving for mur = {inst}')
        for p in [4]:
            print(f'solving for order {p}')
            main(hp=(2,p), geofile='remeshed_matt_key_9_steel.geo')
