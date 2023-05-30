########################################################################
#This code was written by Ben Wilson with the supervison of Paul Ledger
#at the ZCCE, Swansea University
#Powered by NETGEN/NGSolve
########################################################################


#User Inputs

#Geometry
Geometry = "cube.geo"
#(string) Name of the .geo file to be used in the frequency sweep i.e.
# "sphere.geo"


#Scaling to be used in the sweep in meters
alpha = 0.01
#(float) scaling to be applied to the .geo file i.e. if you have defined
#a sphere of unit radius in a .geo file   alpha = 0.01   would simulate a
#sphere with a radius of 0.01m ( or 1cm)


#About the mesh
#How fine should the mesh be
MeshSize = 1
#(int 1-5) this defines how fine the mesh should be for regions that do
#not have maxh values defined for them in the .geo file (1=verycoarse,
#5=veryfine)

#The order of the elements in the mesh
Order = 4
#(int) this defines the order of each of the elements in the mesh 


#About the Frequency sweep (frequencies are in radians per second)
#Minimum frequency (Powers of 10 i.e Start = 2 => 10**2)
Start = 1
#(float)
#Maximum frequency (Powers of 10 i.e Start = 8 => 10**8)
Finish = 6
#(float)
#Number of points in the freqeuncy sweep
Points = 71
#(int) the number of logarithmically spaced points in the sweep

#I only require a single frequency
Single = False
#(boolean) True if single frequency is required
Omega = 100000
#(float) the frequency to be solved if Single = True


#POD
#I want to use POD in the frequency sweep
Pod = False
#(boolean) True if POD is to be used, the number of snapshots can be
#edited in in the Settings.py file


#MultiProcessing
MultiProcessing = True
#(boolean) #I have multiple cores at my disposal and have enough spare RAM
# to run the frequency sweep in parrallel (Edit the number of cores to be
#used in the Settings.py file)








########################################################################


#Main script


#Importing
import sys
import numpy as np
sys.path.insert(0,"Functions")
sys.path.insert(0,"Settings")
from MeshCreation import *
from Settings import *
from SingleSolve import SingleFrequency
from FullSolvers import *
from PODSolvers import *
from ResultsFunctions import *
from Checkvalid import *

if __name__ == '__main__':
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

    #create the array of points to be used in the sweep
    Array = np.logspace(Start,Finish,Points)
#    Array = 2*np.pi*np.array([1.00E+01, 1.26E+01, 1.58E+01, 2.00E+01, 2.51E+01, 3.16E+01, 3.98E+01, 5.01E+01, 6.31E+01, 7.94E+01, 1.00E+02, 1.26E+02, 1.58E+02, 2.00E+02, 2.51E+02, 3.16E+02, 3.98E+02, 5.01E+02, 6.31E+02, 7.94E+02, 1.00E+03, 1.26E+03, 1.58E+03, 2.00E+03, 2.51E+03, 3.16E+03, 3.98E+03, 5.01E+03, 6.31E+03, 7.94E+03, 1.00E+04, 1.26E+04, 1.58E+04, 2.00E+04, 2.51E+04, 3.16E+04, 3.98E+04, 5.01E+04, 6.31E+04, 7.94E+04, 1.00E+05, 1.26E+05, 1.58E+05, 2.00E+05, 2.51E+05, 3.16E+05, 3.98E+05, 5.01E+05, 6.31E+05, 7.94E+05, 1.00E+06, 1.26E+06, 1.58E+06, 2.00E+06, 2.51E+06, 3.16E+06, 3.98E+06, 5.01E+06, 6.31E+06, 7.94E+06, 1.00E+07, 1.26E+07, 1.58E+07, 2.00E+07, 2.51E+07, 3.16E+07, 3.98E+07, 5.01E+07, 6.31E+07, 7.94E+07, 1.00E+08])
#Toykan
    Array = 1000*2*np.pi*np.array([0.11925, 0.178875, 0.2385, 0.298125, 0.35775, 0.477, 0.59625, 0.7155, 0.954, 1.1925, 1.431, 1.908, 2.385, 2.862, 3.816, 4.77, 5.724, 7.632, 9.54, 12.402, 16.218, 20.988, 26.712, 34.344, 43.884, 57.24, 73.458, 95.4])
#John
#    Array = 2*np.pi*1000*np.array([0.01, 0.012589254, 0.015848932, 0.019952623, 0.025118864, 0.031622777, 0.039810717, 0.050118723, 0.063095734, 0.079432823, 0.1, 0.125892541, 0.158489319, 0.199526231, 0.251188643, 0.316227766, 0.398107171, 0.501187234, 0.630957344, 0.794328235, 1, 1.258925412, 1.584893192, 1.995262315, 2.511886432, 3.16227766, 3.981071706, 5.011872336, 6.309573445, 7.943282347, 10, 12.58925412, 15.84893192, 19.95262315, 25.11886432, 31.6227766, 39.81071706, 50.11872336, 63.09573445, 79.43282347, 100, 125.8925412, 158.4893192, 199.5262315, 251.1886432, 316.227766, 398.1071706, 501.1872336, 630.9573445, 794.3282347, 1000])
#Paul
#    Array = 2*np.pi*1000*np.array([0.954, 1.1925, 1.431, 1.908, 2.385, 2.862, 3.816, 4.77, 5.724, 7.632, 9.54, 12.402, 16.218, 20.988, 26.712, 34.344, 43.884, 57.24, 73.458, 95.4])
#    Array = 2*np.pi*np.array([10, 12.58925412, 15.84893192, 19.95262315, 25.11886432, 31.6227766, 39.81071706, 50.11872336, 63.09573445, 79.43282347, 100, 125.8925412, 158.4893192, 199.5262315, 251.1886432, 316.227766, 398.1071706, 501.1872336, 630.9573445, 794.3282347, 1000, 1258.925412, 1584.893192, 1995.262315, 2511.886432, 3162.27766, 3981.071706, 5011.872336, 6309.573445, 7943.282347, 10000, 12589.25412, 15848.93192, 19952.62315, 25118.86432, 31622.7766, 39810.71706, 50118.72336, 63095.73445, 79432.82347, 100000, 125892.5412, 158489.3192, 199526.2315, 251188.6432, 316227.766, 398107.1706, 501187.2336, 630957.3445, 794328.2347, 1000000, 1258925.412, 1584893.192, 1995262.315, 2511886.432, 3162277.66, 3981071.706, 5011872.336, 6309573.445, 7943282.347, 10000000, 12589254.12, 15848931.92, 19952623.15, 25118864.32, 31622776.6, 39810717.06, 50118723.36, 63095734.45, 79432823.47, 100000000])
#    Array = 2*np.pi*np.array([1.0E+07])
    PlotPod, PODErrorBars, EddyCurrentTest, vtk_output, Refine = AdditionalOutputs()
    SavePOD = False
    if PODErrorBars!=True:
        ErrorTensors=False
    else:
        ErrorTensors=True
    PODArray = np.logspace(Start,Finish,PODPoints)
#    PODArray = 2*np.pi*np.array([1.00000000e+01, 3.83118685e+01, 1.46779927e+02, 5.62341325e+02, 2.15443469e+03, 8.25404185e+03, 3.16227766e+04, 1.21152766e+05, 4.64158883e+05, 1.77827941e+06, 6.81292069e+06, 2.61015722e+07, 1.00000000e+08])
#    PODArray = 1000*2*np.pi*np.array([0.11925, 0.178875, 0.2385, 0.298125, 0.35775, 0.477, 0.59625, 0.7155, 0.954, 1.1925, 1.431, 1.908, 2.385, 2.862, 3.816, 4.77, 5.724, 7.632, 9.54, 12.402, 16.218, 20.988, 26.712, 34.344, 43.884, 57.24, 73.458, 95.4])
#    PODArray = 2*np.pi*np.logspace(2.0764583877121514,4.979548374704095,13)

    #Create the folders which will be used to save everything
    sweepname = FolderMaker(Geometry, Single, Array, Omega, Pod, PlotPod, PODArray, PODTol, alpha, Order, MeshSize, mur, sig, ErrorTensors, vtk_output)



    #Run the sweep
    
    #Check the validity of the eddy-current model for the object
    if EddyCurrentTest == True:
        EddyCurrentTest = Checkvalid(Geometry,Order,alpha,inorout,mur,sig)

    if Single==True:
        if MultiProcessing!=True:
            CPUs = 1
        MPT, EigenValues, N0, elements = SingleFrequency(Geometry,Order,alpha,inorout,mur,sig,Omega,CPUs,vtk_output,Refine)
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
        else:
            if MultiProcessing==True:
                TensorArray, EigenValues, N0, elements = FullSweepMulti(Geometry,Order,alpha,inorout,mur,sig,Array,CPUs,BigProblem)
            else:
                TensorArray, EigenValues, N0, elements = FullSweep(Geometry,Order,alpha,inorout,mur,sig,Array,BigProblem)
    

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



print("End of Program")
