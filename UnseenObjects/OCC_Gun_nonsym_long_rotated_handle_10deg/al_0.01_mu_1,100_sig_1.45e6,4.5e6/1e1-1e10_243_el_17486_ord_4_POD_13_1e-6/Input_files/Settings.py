#This file contains functions which allow the user to edit default parameters
#Importing
import numpy as np
from ngsolve import *

# from ngsolve import ngsglobals
# ngsglobals.msg_level = 0

#Function definition to set up default settings
def DefaultSettings():
    #How many cores to be used (monitor memory consuption)
    CPUs = 1
    #(int)

    #Is it a big problem (more memory efficiency but slower)
    BigProblem = False
    #(boolean)

    #How many snapshots should be taken
    PODPoints = 13
    #(int)

    #Tolerance to be used in the TSVD
    PODTol = 10**-6
    #(float)

    #Use an old mesh
    OldMesh = False
    #(boolean) Note that this still requires the relavent .geo file to obtain
    #information about the materials in the mesh

    #Use an old POD model saved to disk. This allows the user to specify a new set of frequencies without recomputing
    #the POD snapshots.
    OldPOD = False
    #(boolean)

    # Number of parallel threads to be used when constructing linear and bilinear forms, and performing the iterative
    # solve of theta1. Set to 'default' to use all available threads.
    NumSolverThreads = 'default'
    # (int)

    # The tolerance to be used for dropping near zero values when creating large matrices (including interior dofs)
    # for MPT and POD calculations
    drop_tol = None #1e-12 # None
    # A smaller value will reduce the sparsity and increase the memory requirements. If set too small
    # this may lead to excessive matrix fill in and large memory useage at high p especially if prismatic layers
    # are used. Using drop_tol = None no values will be dropped.

    return CPUs,BigProblem,PODPoints,PODTol,OldMesh, OldPOD, NumSolverThreads, drop_tol

def AdditionalOutputs():
    #Plot the POD points
    PlotPod = True
    #(boolean) do you want to plot the snapshots (This requires additional
    #calculations and will slow down sweep by around 2% for default settings)

    #Produce certificate bounds for POD outputs
    PODErrorBars = False
    #(boolean)

    #Test where the eddy-current model breaks for the object
    EddyCurrentTest = False
    #(boolean)

    #Produce a vtk outputfile for the eddy-currents (outputs a large file!)
    vtk_output = False
    #(boolean) do you want to produce a vtk file of the eddy currents in the
    #object (single frequency only)

    #Refine the vtk output (extremely large file!)
    Refine_vtk = True
    #(boolean) do you want ngsolve to refine the solution before exporting
    #to the vtk file (single frequency only)
    #(not compatable with all NGSolve versions)

    # Save out the left singular vector in POD modes (Very large files!)
    Save_U = False
    # (boolean) option to save out the left singular vector used by the POD. This would allow the user to restart the
    # POD operation, by setting OLDPOD=True, without needing to recalculate the full order snapshot solutions.

    return PlotPod, PODErrorBars, EddyCurrentTest, vtk_output, Refine_vtk, Save_U

def PrerunCheckSettings():
    """
    James Elgy 2023:
    Settings for prerun checks that are applied before running large sweep.
    """

    # (bool) Flag for running the checks. Recommended that this is True for first run.
    run = True

    # (float) tolerance required for the bilinear forms to be considered converged.
    bilinear_tol = 1e-10

    # (int) maximum iterations for the iterative bilinear form convergence check. The maximum bonus integration order
    # applied to the bilinear forms is 2*(max_iter-1). Large number of iterations can take some time.
    max_iter = 10

    return run, bilinear_tol, max_iter

#Function definition to set up default settings
def SaverSettings():
    #Place to save the results to
    FolderName = "Default" 
    FolderName = 'GunSims'
    # FolderName = '/run/user/1001/gvfs/sftp:host=10.50.20.1,user=james/home/james/'
    # FolderName = r'james@10.50.20.1/home/james'
    #"/home/paul/Coding/PaulResults/PaulMPTResults/Results/" 
    
    #(string) This defines the folder (and potentially subfolders) the
    #data will be saved in (if "Default" then a predetermined the data
    #will be saved in a predetermined folder structure)
    #Example input "MyShape/MyFrequencySweep"

    return FolderName



#Function definition to set up parameters relating to solving the problems
def SolverParameters():
    #Parameters associated with solving the problem can edit this
    #preconditioner to be used
    Solver = "bddc"
    #(string) "bddc"/"local"

    #regularisation
    epsi = 10**-10
    #(float) regularisation to be used in the problem

    #Maximum iterations to be used in solving the problem
    Maxsteps = 1500
    #(int) maximum number of iterations to be used in solving the problem
    #the bddc will converge in most cases in less than 200 iterations
    #the local will take more

    #Relative tolerance
    Tolerance = 10**-8
    #(float) the amount the redsidual must decrease by relatively to solve
    #the problem

    #Additional Integrarion Order
    AdditionalIntFactor = 0
    #(int) Additional integration order (AdditionalIntFactor) to be used with SymbolicBFI and SymbolicLFI.
    # This is introduced to account for under integration in the bilinear forms and linear forms used in the faster
    # matrix multiplication used for the POD method and get agreement between integration and matrix multiplication
    # methods. Setting this to 0 will remove the effect. Only used for calculating tensor coeffs in POD mode.

    #Method for calculating tensor coefficieints in POD mode.
    use_integral = False

    #print convergence of the problem
    ngsglobals.msg_level = 0
    #(int) Do you want information about the solving of the problems
    #Suggested inputs
    #0 for no information, 3 for information of convergence
    #Other useful options 1,6
    return Solver,epsi,Maxsteps,Tolerance, AdditionalIntFactor, use_integral


def IterativePODParameters():
    """
    James Elgy - 2022:
    Settings for the Iterative POD method used by PODSolvers.IterativePOD().
    Returns
    -------
    NAdditionalSnapshotsPerIter (int) number of additional snapshots to compute per iteration.
    MaxIter (int) total number of iterations to consider.
    Tol (float) tolerance of maxerror/object_volume for terminating the iterative process.
    PlotUpdatedPOD (bool) option to plot error certificates and tensor coefficients at each iteration.
    """

    # (int) Number of additional snapshots to add per iteration.
    NAdditionalSnapshotsPerIter = 2

    # (int) Maximum number of iterations that will run in the iterative process.
    MaxIter = 20

    # (float) Stopping tolerance (max(error)/object_volume) for the iterative process.
    Tol = 1e-3

    # (bool) Option to also plot out the tensor coefficients and error certificates for each iteration.
    PlotUpdatedPOD = True

    return NAdditionalSnapshotsPerIter, MaxIter, Tol, PlotUpdatedPOD
