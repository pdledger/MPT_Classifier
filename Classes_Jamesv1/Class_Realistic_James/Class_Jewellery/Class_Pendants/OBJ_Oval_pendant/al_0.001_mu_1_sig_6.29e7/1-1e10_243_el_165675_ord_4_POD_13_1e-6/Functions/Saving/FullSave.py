import os
import sys
from math import floor, log10
import numpy as np
from shutil import copyfile
from zipfile import *

import netgen.meshing as ngmeshing
from ngsolve import Mesh

sys.path.insert(0,"Functions")
from Settings import SaverSettings
from .EigPlotter import *
from .TensorPlotter import *
from .ErrorPlotter import *
from .FtoS import *
from .DictionaryList import *


def FullSave(Geometry, Array, TensorArray, EigenValues, N0, Pod, PODArray, PODTol, elements, alpha, Order, MeshSize,
             mur, sig, ErrorTensors, EddyCurrentTest, invariants):
    """
    B.A. Wilson, J.Elgy, P.D. Ledger.2020-2023.
    Save data and make folder structure.

    Args:
        Geometry (str): geometry file name. E.g. 'sphere'
        Array (list): array of frequencies in sweep
        TensorArray (np.ndarray): Nx9 complex tensor coefficients.
        EigenValues (np.ndarray): Nx3 complex eigenvalues
        N0 (np.ndarray): 3x3 N0 coefficient
        Pod (bool): bool for if sweep used POD
        PODArray (list | np.ndarray): list of K frequencies (rad/s) for POD snapshots.
        PODTol (float): Tolerance for truncated SVD
        elements (int): number of elements in mesh
        alpha (float): object size scaling
        Order (int): order of finite element space.
        MeshSize (float): No longer used. Originally this was max element size.
        mur (dict): dictionary of mur in each region
        sig (dict): dictionary of sigma in each region
        ErrorTensors (np.ndarray): Nx6 array of error certificates. Note that errors are symmetric so we only store upper trianglar part.
        EddyCurrentTest (float | None): max frequency for eddy current regime, or None if not calculated.
        invariants (np.ndarray): Nx3 MPT Tensor invarients.
    """
    
    
    # Find how the user wants the data to be saved
    # FolderStructure = SaverSettings()

    # Create a temp folder in the results directory.
    FolderStructure = 'Default'

    if FolderStructure == "Default":
        # Create the file structure
        # Define constants for the folder name
        objname = Geometry[:-4]
        minF = Array[0]
        strminF = FtoS(minF)
        maxF = Array[-1]
        strmaxF = FtoS(maxF)
        Points = len(Array)
        PODPoints = len(PODArray)
        strmur = DictionaryList(mur, False)
        strsig = DictionaryList(sig, True)
        strPODTol = FtoS(PODTol)

        # Define the main folder structure
        subfolder1 = "al_" + str(alpha) + "_mu_" + strmur + "_sig_" + strsig
        if Pod == True:
            subfolder2 = strminF + "-" + strmaxF + "_" + str(Points) + "_el_" + str(elements) + "_ord_" + str(
                Order) + "_POD_" + str(PODPoints) + "_" + strPODTol
        else:
            subfolder2 = strminF + "-" + strmaxF + "_" + str(Points) + "_el_" + str(elements) + "_ord_" + str(Order)
        sweepname = objname + "/" + subfolder1 + "/" + subfolder2
    else:
        sweepname = FolderStructure

    # Save the data
    np.savetxt("Results/" + sweepname + "/Data/Frequencies.csv", Array, delimiter=",")
    np.savetxt("Results/" + sweepname + "/Data/Eigenvalues.csv", EigenValues, delimiter=",")
    np.savetxt("Results/" + sweepname + "/Data/N0.csv", N0, delimiter=",")
    np.savetxt("Results/" + sweepname + "/Data/Tensors.csv", TensorArray, delimiter=",")
    np.savetxt("Results/" + sweepname + "/Data/Invariants.csv", invariants, delimiter=",")

    if Pod == True:
        np.savetxt("Results/" + sweepname + "/Data/PODFrequencies.csv", PODArray, delimiter=",")
    if isinstance(EddyCurrentTest, float):
        f = open('Results/' + sweepname + '/Data/Eddy-current_breakdown.txt', 'w+')
        f.write('omega = ' + str(round(EddyCurrentTest)))
        f.close()

    # Format the tensor arrays so they can be plotted
    PlottingTensorArray = np.zeros([Points, 6], dtype=complex)
    PlottingTensorArray = np.concatenate(
        [np.concatenate([TensorArray[:, :3], TensorArray[:, 4:6]], axis=1), TensorArray[:, 8:9]], axis=1)
    try:
        ErrorTensors[:, [1, 3]] = ErrorTensors[:, [3, 1]]
        ErrorTensors[:, [2, 4]] = ErrorTensors[:, [4, 2]]
        ErrorTensors[:, [4, 5]] = ErrorTensors[:, [5, 4]]
    except:
        pass

    # Define where to save the graphs
    savename = "Results/" + sweepname + "/Graphs/"

    # Plot the graphs
    Show = EigPlotter(savename, Array, EigenValues, EddyCurrentTest)

    if Pod == True:
        try:
            if ErrorTensors == False:
                Show = TensorPlotter(savename, Array, PlottingTensorArray, EddyCurrentTest)
        except:
            Show = ErrorPlotter(savename, Array, PlottingTensorArray, ErrorTensors, EddyCurrentTest)
            # Change the format of the error bars to the format of the Tensors
            Errors = np.zeros([Points, 9])
            Errors[:, 0] = ErrorTensors[:, 0]
            Errors[:, 1] = ErrorTensors[:, 3]
            Errors[:, 2] = ErrorTensors[:, 4]
            Errors[:, 3] = ErrorTensors[:, 3]
            Errors[:, 4] = ErrorTensors[:, 1]
            Errors[:, 5] = ErrorTensors[:, 5]
            Errors[:, 6] = ErrorTensors[:, 4]
            Errors[:, 7] = ErrorTensors[:, 5]
            Errors[:, 8] = ErrorTensors[:, 2]
            np.savetxt("Results/" + sweepname + "/Data/ErrorBars.csv", Errors, delimiter=",")
    else:
        Show = TensorPlotter(savename, Array, PlottingTensorArray, EddyCurrentTest)

    # plot the graph if required
    if Show == True:
        plt.show()

    return
