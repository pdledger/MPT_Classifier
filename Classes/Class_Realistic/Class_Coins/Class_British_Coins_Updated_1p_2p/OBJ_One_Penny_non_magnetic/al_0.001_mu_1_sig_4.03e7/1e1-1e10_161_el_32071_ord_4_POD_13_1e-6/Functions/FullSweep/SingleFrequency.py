# This file contains the function called from the main.py file when Single=True
# Functions -SingleFrequency (Solve for one value of omega)
# Importing
import os
import sys
import time
import multiprocessing as multiprocessing
import tqdm.auto as tqdm
import cmath
import numpy as np

import netgen.meshing as ngmeshing
from ngsolve import *

sys.path.insert(0, "Functions")
from ..Core_MPT.Theta1 import *
from ..Core_MPT.Theta0 import *
from ..Core_MPT.MPTCalculator import *
from ..Core_MPT.imap_execution import *
from ..Saving.FtoS import *
from ..Saving.DictionaryList import *
from ..Core_MPT.MPT_Preallocation import *
from ..Core_MPT.Solve_Theta_0_Problem import *

sys.path.insert(0, "Settings")
from Settings import SolverParameters
import gc

from Functions.Helper_Functions.count_prismatic_elements import count_prismatic_elements


def SingleFrequency(Object, Order, alpha, inorout, mur, sig, Omega, CPUs, VTK, Refine, Integration_Order, Additional_Int_Order, Order_L2, sweepname,
                    curve=5, theta_solutions_only=False, num_solver_threads='default'):

    _, Mu0, _, _, _, _, inout, mesh, mu, numelements, sigma, _ = MPT_Preallocation([Omega], Object, [], curve, inorout, mur, sig, Order, 0, sweepname)
    # Set up the Solver Parameters
    Solver, epsi, Maxsteps, Tolerance, _, use_integral = SolverParameters()

    # Set up how the tensors will be stored
    N0 = np.zeros([3, 3])
    R = np.zeros([3, 3])
    I = np.zeros([3, 3])

    #########################################################################
    # Theta0
    # This section solves the Theta0 problem to calculate both the inputs for
    # the Theta1 problem and calculate the N0 tensor
    # Here, we have set the solver not to use the recovery mode.
    Theta0Sol, Theta0i, Theta0j, fes, ndof, evec = Solve_Theta_0_Problem(Additional_Int_Order, CPUs, Maxsteps, Order,
                                                                         Solver,
                                                                         Tolerance, alpha, epsi, inout, mesh, mu,
                                                                         False, '')

    # Calculate the N0 tensor
    VolConstant = Integrate(1 - mu ** (-1), mesh, order=Integration_Order)
    for i in range(3):
        Theta0i.vec.FV().NumPy()[:] = Theta0Sol[:, i]
        for j in range(3):
            Theta0j.vec.FV().NumPy()[:] = Theta0Sol[:, j]
            if i == j:
                N0[i, j] = (alpha ** 3) * (VolConstant + (1 / 4) * (
                    Integrate(mu ** (-1) * (InnerProduct(curl(Theta0i), curl(Theta0j))), mesh, order=Integration_Order)))
            else:
                N0[i, j] = (alpha ** 3 / 4) * (
                    Integrate(mu ** (-1) * (InnerProduct(curl(Theta0i), curl(Theta0j))), mesh, order=Integration_Order))

    # Removing gradient terms introduced by making fes the same size as fes2:
    # Poission Projection to account for gradient terms:
    u, v = fes.TnT()
    m = BilinearForm(fes)
    m += u * v * dx
    m.Assemble()

    # build gradient matrix as sparse matrix (and corresponding scalar FESpace)
    gradmat, fesh1 = fes.CreateGradient()

    gradmattrans = gradmat.CreateTranspose()  # transpose sparse matrix
    math1 = gradmattrans @ m.mat @ gradmat  # multiply matrices
    math1[0, 0] += 1  # fix the 1-dim kernel
    invh1 = math1.Inverse(inverse="sparsecholesky")

    # build the Poisson projector with operator Algebra:
    proj = IdentityMatrix() - gradmat @ invh1 @ gradmattrans @ m.mat
    theta0 = GridFunction(fes)
    for i in range(3):
        theta0.vec.FV().NumPy()[:] = Theta0Sol[:, i]
        theta0.vec.data = proj * (theta0.vec)
        Theta0Sol[:, i] = theta0.vec.FV().NumPy()[:]

    #########################################################################
    # Theta1
    # This section solves the Theta1 problem and saves the solution vectors

    # Setup the finite element space
    dom_nrs_metal = [0 if mat == "air" else 1 for mat in mesh.GetMaterials()]
    fes2 = HCurl(mesh, order=Order, dirichlet="outer", complex=True, gradientdomains=dom_nrs_metal)
    # Count the number of degrees of freedom
    ndof2 = fes2.ndof

    # Define the vectors for the right hand side
    xivec = [CoefficientFunction((0, -z, y)), CoefficientFunction((z, 0, -x)), CoefficientFunction((-y, x, 0))]

    # Setup the array which will be used to store the solution vectors
    Theta1Sol = np.zeros([ndof2, 3], dtype=complex)

    # Set up the inputs for the problem
    Runlist = []
    nu = Omega * Mu0 * (alpha ** 2)
    for i in range(3):
        if CPUs < 3:
            NewInput = (
            fes, fes2, Theta0Sol[:, i], xivec[i], Order, alpha, nu, sigma, mu, inout, Tolerance, Maxsteps, epsi, Omega,
            i + 1, 3, Solver, num_solver_threads, Additional_Int_Order, 'Theta1')
        else:
            NewInput = (
            fes, fes2, Theta0Sol[:, i], xivec[i], Order, alpha, nu, sigma, mu, inout, Tolerance, Maxsteps, epsi, Omega,
            "No Print", 3, Solver, num_solver_threads, Additional_Int_Order, 'Theta1')
        Runlist.append(NewInput)

    # Run on the multiple cores
    with multiprocessing.Pool(CPUs) as pool:
        Output = list(tqdm.tqdm(pool.map(imap_version, Runlist), total=len(Runlist), desc='Solving Theta1'))
        # Output = pool.starmap(Theta1, Runlist)
    print(' solved theta1 problem       ')

    # Unpack the outputs
    for i, OutputNumber in enumerate(Output):
        Theta1Sol[:, i] = OutputNumber

    if theta_solutions_only == True:
        return Theta0Sol, Theta1Sol

    # Create the VTK output if required
    if VTK == True:
        print(' creating vtk output', end='\r')
        ThetaE1 = GridFunction(fes2)
        ThetaE2 = GridFunction(fes2)
        ThetaE3 = GridFunction(fes2)
        ThetaE1.vec.FV().NumPy()[:] = Output[0]
        ThetaE2.vec.FV().NumPy()[:] = Output[1]
        ThetaE3.vec.FV().NumPy()[:] = Output[2]
        E1Mag = CoefficientFunction(
            sqrt(InnerProduct(ThetaE1.real, ThetaE1.real) + InnerProduct(ThetaE1.imag, ThetaE1.imag)))
        E2Mag = CoefficientFunction(
            sqrt(InnerProduct(ThetaE2.real, ThetaE2.real) + InnerProduct(ThetaE2.imag, ThetaE2.imag)))
        E3Mag = CoefficientFunction(
            sqrt(InnerProduct(ThetaE3.real, ThetaE3.real) + InnerProduct(ThetaE3.imag, ThetaE3.imag)))
        Sols = []
        Sols.append(dom_nrs_metal)
        Sols.append((ThetaE1 * 1j * Omega * sigma).real)
        Sols.append((ThetaE1 * 1j * Omega * sigma).imag)
        Sols.append((ThetaE2 * 1j * Omega * sigma).real)
        Sols.append((ThetaE2 * 1j * Omega * sigma).imag)
        Sols.append((ThetaE3 * 1j * Omega * sigma).real)
        Sols.append((ThetaE3 * 1j * Omega * sigma).imag)
        Sols.append(E1Mag * Omega * sigma)
        Sols.append(E2Mag * Omega * sigma)
        Sols.append(E3Mag * Omega * sigma)

        # Creating Save Name:
        strmur = DictionaryList(mur, False)
        strsig = DictionaryList(sig, True)
        savename = "Results/" + Object[:-4] + f"/al_{alpha}_mu_{strmur}_sig_{strsig}" + "/om_" + FtoS(Omega) + f"_el_{numelements}_ord_{Order}/Data/"
        if Refine == True:
            vtk = VTKOutput(ma=mesh, coefs=Sols,
                            names=["Object", "E1real", "E1imag", "E2real", "E2imag", "E3real", "E3imag", "E1Mag",
                                   "E2Mag", "E3Mag"], filename=savename + Object[:-4], subdivision=3)
        else:
            vtk = VTKOutput(ma=mesh, coefs=Sols,
                            names=["Object", "E1real", "E1imag", "E2real", "E2imag", "E3real", "E3imag", "E1Mag",
                                   "E2Mag", "E3Mag"], filename=savename + Object[:-4], subdivision=0)
        vtk.Do()

        # Compressing vtk output and sending to zip file:
        zipObj = ZipFile(savename + 'VTU.zip', 'w', ZIP_DEFLATED)
        zipObj.write(savename + Object[:-4] + '.vtu', os.path.basename(savename + Object[:-4] + '.vtu'))
        zipObj.close()
        os.remove(savename + Object[:-4] + '.vtu')
        print(' vtk output created     ')

    #########################################################################
    # Calculate the tensor and eigenvalues

    # Create the inputs for the calculation of the tensors
    print(' calculating the tensor  ', end='\r')
    Runlist = []
    nu = Omega * Mu0 * (alpha ** 2)
    R, I = MPTCalculator(mesh, fes, fes2, Theta1Sol[:, 0], Theta1Sol[:, 1], Theta1Sol[:, 2], Theta0Sol, xivec, alpha,
                         mu, sigma, inout, nu, "No Print", 1, Order, Integration_Order)
    print(' calculated the tensor             ')

    # Unpack the outputs
    MPT = N0 + R + 1j * I
    RealEigenvalues = np.sort(np.linalg.eigvals(N0 + R))
    ImaginaryEigenvalues = np.sort(np.linalg.eigvals(I))
    EigenValues = RealEigenvalues + 1j * ImaginaryEigenvalues

    # del Theta1i, Theta1j, Theta0i, Theta0j, fes, fes2, Theta0Sol, Theta1Sol
    gc.collect()

    return MPT, EigenValues, N0, numelements, (ndof, ndof2)
