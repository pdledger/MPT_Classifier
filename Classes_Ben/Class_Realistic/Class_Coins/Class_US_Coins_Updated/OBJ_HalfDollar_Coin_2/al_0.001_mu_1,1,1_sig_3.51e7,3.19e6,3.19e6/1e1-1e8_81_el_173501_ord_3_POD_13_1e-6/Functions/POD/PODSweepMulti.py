"""
Edit 06 Aug 2022: James Elgy
Changed how N0 was calculated for PODSweep to be consistent with PODSweepMulti.
Changed pool generation to spawn to fix linux bug.

"""
#Importing
import gc
import os
from contextlib import contextmanager
import sys
import time
import math
import multiprocessing as multiprocessing
multiprocessing.freeze_support()
import warnings
from warnings import warn
from tqdm import tqdm
import cmath
import numpy as np
import scipy.signal
import scipy.sparse as sp
import scipy.sparse.linalg as spl

import netgen.meshing as ngmeshing
from ngsolve import *

sys.path.insert(0,"Functions")
from ..Core_MPT.Theta0 import *
from ..Core_MPT.Theta1 import *
from ..Core_MPT.Theta1_Sweep import *
from ..Core_MPT.Theta1_Lower_Sweep import *
from ..Core_MPT.Theta1_Lower_Sweep_Mat_Method import *
from ..Core_MPT.MPT_Preallocation import *
from ..POD.calc_error_certificates import *
from ..Core_MPT.imap_execution import *
from ..Core_MPT.supress_stdout import *
#from ..FullSweep.generate_VTK import *
sys.path.insert(0,"Settings")
from Settings import SolverParameters, DefaultSettings, IterativePODParameters

# Importing matplotlib for plotting comparisons
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
#from .test_comparison import run_test_comparison


def PODSweepMulti(Object, Order, alpha, inorout, mur, sig, Array, PODArray, PODTol, PlotPod, CPUs, sweepname, SavePOD,
                  PODErrorBars, BigProblem, Integration_Order, Additional_Int_Order, curve=5, recoverymode=False, NumSolverThreads='default', save_U=False):
    timing_dictionary = {}

    timing_dictionary['start_time'] = time.time()

    EigenValues, Mu0, N0, NumberofFrequencies, NumberofSnapshots, TensorArray,  inout, mesh, mu, numelements, sigma = MPT_Preallocation(
        Array, Object, PODArray, curve, inorout, mur, sig)
    # Set up the Solver Parameters
    Solver, epsi, Maxsteps, Tolerance, _, use_integral = SolverParameters()

    #########################################################################
    # Theta0
    # This section solves the Theta0 problem to calculate both the inputs for
    # the Theta1 problem and calculate the N0 tensor

    # Setup the finite element space
    dom_nrs_metal = [0 if mat == "air" else 1 for mat in mesh.GetMaterials()]
    fes = HCurl(mesh, order=Order, dirichlet="outer", gradientdomains=dom_nrs_metal)
    # fes = HCurl(mesh, order=Order, dirichlet="outer", flags = { "nograds" : True })
    # Count the number of degrees of freedom
    ndof = fes.ndof

    # Define the vectors for the right hand side
    evec = [CoefficientFunction((1, 0, 0)), CoefficientFunction((0, 1, 0)), CoefficientFunction((0, 0, 1))]

    # Setup the grid functions and array which will be used to save
    Theta0i = GridFunction(fes)
    Theta0j = GridFunction(fes)
    Theta0Sol = np.zeros([ndof, 3])

    if recoverymode is False:
        # Setup the inputs for the functions to run
        Theta0CPUs = min(3, multiprocessing.cpu_count(), CPUs)
        Runlist = []
        for i in range(3):
            if Theta0CPUs < 3:
                NewInput = (fes, Order, alpha, mu, inout, evec[i], Tolerance, Maxsteps, epsi, i + 1, Solver, Additional_Int_Order, 'Theta0')
            else:
                NewInput = (fes, Order, alpha, mu, inout, evec[i], Tolerance, Maxsteps, epsi, "No Print", Solver, Additional_Int_Order, 'Theta0')
            Runlist.append(NewInput)
        # Run on the multiple cores
        with multiprocessing.get_context("spawn").Pool(Theta0CPUs) as pool:
            Output = list(tqdm.tqdm(pool.map(imap_version, Runlist), total=len(Runlist), desc='Solving Theta0'))
            # Output = pool.starmap(Theta0, Runlist)
        print(' solved theta0 problems    ')

        # Unpack the outputs
        for i, Direction in enumerate(Output):
            Theta0Sol[:, i] = Direction
    else:
        try:
            Theta0Sol = np.load('Results/' + sweepname + '/Data/Theta0.npy')
        except FileNotFoundError:
            warn(
                'Could not find theta0 file at:' + ' Results/' + sweepname + '/Data/Theta0.npy \nFalling back to calculation of theta0')
            # Setup the inputs for the functions to run
            Theta0CPUs = min(3, multiprocessing.cpu_count(), CPUs)
            Runlist = []
            for i in range(3):
                if Theta0CPUs < 3:
                    NewInput = (fes, Order, alpha, mu, inout, evec[i], Tolerance, Maxsteps, epsi, i + 1, Solver, Additional_Int_Order)
                else:
                    NewInput = (fes, Order, alpha, mu, inout, evec[i], Tolerance, Maxsteps, epsi, "No Print", Solver, Additional_Int_Order)
                Runlist.append(NewInput)
            # Run on the multiple cores
            with multiprocessing.get_context("spawn").Pool(Theta0CPUs) as pool:
                Output = pool.starmap(Theta0, Runlist)
            print(' solved theta0 problems    ')

            # Unpack the outputs
            for i, Direction in enumerate(Output):
                Theta0Sol[:, i] = Direction

    if recoverymode is False:
        np.save('Results/' + sweepname + '/Data/Theta0', Theta0Sol)


    # Poission Projection to acount for gradient terms:
    u, v = fes.TnT()
    m = BilinearForm(fes)
    m += SymbolicBFI(u * v, bonus_intorder=Additional_Int_Order)
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

    timing_dictionary['Theta0'] = time.time()

    #########################################################################
    # Theta1
    # This section solves the Theta1 problem and saves the solution vectors

    print(' solving theta1 snapshots')
    # Setup the finite element space
    dom_nrs_metal = [0 if mat == "air" else 1 for mat in mesh.GetMaterials()]
    fes2 = HCurl(mesh, order=Order, dirichlet="outer", complex=True, gradientdomains=dom_nrs_metal)
    # Count the number of degrees of freedom
    ndof2 = fes2.ndof

    # Define the vectors for the right hand side
    xivec = [CoefficientFunction((0, -z, y)), CoefficientFunction((z, 0, -x)), CoefficientFunction((-y, x, 0))]

    if recoverymode is False:
        # Work out where to send each frequency
        Theta1_CPUs = min(NumberofSnapshots, multiprocessing.cpu_count(), CPUs)
        Core_Distribution = []
        Count_Distribution = []
        for i in range(Theta1_CPUs):
            Core_Distribution.append([])
            Count_Distribution.append([])

        # Distribute between the cores
        CoreNumber = 0
        count = 1
        for i, Omega in enumerate(PODArray):
            Core_Distribution[CoreNumber].append(Omega)
            Count_Distribution[CoreNumber].append(i)
            if CoreNumber == CPUs - 1 and count == 1:
                count = -1
            elif CoreNumber == 0 and count == -1:
                count = 1
            else:
                CoreNumber += count

        # Create the inputs
        Runlist = []
        manager = multiprocessing.Manager()
        counter = manager.Value('i', 0)

        for i in range(len(PODArray)):
            if PlotPod == True:
                Runlist.append((np.asarray([PODArray[i]]), mesh, fes, fes2, Theta0Sol, xivec, alpha, sigma, mu, inout,
                                Tolerance, Maxsteps, epsi, Solver, N0, NumberofSnapshots, True, True, counter,
                                BigProblem, Order, NumSolverThreads,Integration_Order, Additional_Int_Order, 'Theta1_Sweep'))
            else:
                Runlist.append((np.asarray([PODArray[i]]), mesh, fes, fes2, Theta0Sol, xivec, alpha, sigma, mu, inout,
                                Tolerance, Maxsteps, epsi, Solver, N0, NumberofSnapshots, True, False, counter,
                                BigProblem, Order, NumSolverThreads, Integration_Order, Additional_Int_Order, 'Theta1_Sweep'))

        # Run on the multiple cores
        multiprocessing.freeze_support()
        tqdm.tqdm.set_lock(multiprocessing.RLock())
        if ngsglobals.msg_level != 0:
            to = os.devnull
        else:
            to = os.devnull
        with supress_stdout(to=to):
            with multiprocessing.get_context("spawn").Pool(Theta1_CPUs, maxtasksperchild=1, initializer=tqdm.tqdm.set_lock, initargs=(tqdm.tqdm.get_lock(),)) as pool:
                Outputs = list(tqdm.tqdm(pool.imap(imap_version, Runlist, chunksize=1), total=len(Runlist), desc='Solving Theta1 Snapshots', dynamic_ncols=True,
                                         position=0, leave=True))

        try:
            pool.terminate()
            print('manually closed pool')
        except:
            print('Pool has already closed.')

        # Unpack the results
        if BigProblem == True:
            Theta1Sols = np.zeros([ndof2, NumberofSnapshots, 3], dtype=np.complex64)
        else:
            Theta1Sols = np.zeros([ndof2, NumberofSnapshots, 3], dtype=complex)
        if PlotPod == True:
            PODTensors = np.zeros([NumberofSnapshots, 9], dtype=complex)
            PODEigenValues = np.zeros([NumberofSnapshots, 3], dtype=complex)

        for i in range(len(Outputs)):
            if PlotPod is True:
                PODEigenValues[i, :] = Outputs[i][1][0]
                PODTensors[i, :] = Outputs[i][0][0]
                for j in range(ndof2):
                    Theta1Sols[j,i,:] = Outputs[i][2][j][0]
            else:
                for j in range(ndof2):
                    Theta1Sols[j, i, :] = Outputs[i][0][j][0]

        timing_dictionary['Theta1'] = time.time()

        ########################################################################
        # Create the ROM

        #########################################################################
        # POD

        print(' performing SVD              ', end='\r')
        # Perform SVD on the solution vector matrices
        u1Truncated, s1, vh1 = np.linalg.svd(Theta1Sols[:, :, 0], full_matrices=False)
        u2Truncated, s2, vh2 = np.linalg.svd(Theta1Sols[:, :, 1], full_matrices=False)
        u3Truncated, s3, vh3 = np.linalg.svd(Theta1Sols[:, :, 2], full_matrices=False)
        # Get rid of the solution vectors
        Theta1Sols = None
        # Print an update on progress
        print(' SVD complete      ')

        # scale the value of the modes
        s1norm = s1 / s1[0]
        s2norm = s2 / s2[0]
        s3norm = s3 / s3[0]

        # Decide where to truncate
        cutoff = NumberofSnapshots
        for i in range(NumberofSnapshots):
            if s1norm[i] < PODTol:
                if s2norm[i] < PODTol:
                    if s3norm[i] < PODTol:
                        cutoff = i
                        break

        # Truncate the SVD matrices
        u1Truncated = u1Truncated[:, :cutoff]
        u2Truncated = u2Truncated[:, :cutoff]
        u3Truncated = u3Truncated[:, :cutoff]

        print(f' Number of retained modes = {cutoff}')
        plt.figure()
        plt.semilogy(s1norm, label=f'$i={1}$')
        plt.semilogy(s2norm, label=f'$i={2}$')
        plt.semilogy(s3norm, label=f'$i={3}$')
        plt.xlabel('Mode')
        plt.ylabel('Normalised Signular Values')
        plt.legend()
        plt.savefig('Results/' + sweepname + '/Graphs/SVD_Decay.pdf')

    else:
        print(' Loading truncated vectors')
        # Loading in Left Singular Vectors:
        u1Truncated = np.load('Results/' + sweepname + '/Data/U1_truncated.npy')
        u2Truncated = np.load('Results/' + sweepname + '/Data/U2_truncated.npy')
        u3Truncated = np.load('Results/' + sweepname + '/Data/U3_truncated.npy')
        try:
            PODTensors = np.genfromtxt('Results/' + sweepname + '/Data/PODTensors.csv', dtype=complex, delimiter=',')
            PODEigenValues = np.genfromtxt('Results/' + sweepname + '/Data/PODEigenvalues.csv', dtype=complex,
                                           delimiter=',')
        except FileNotFoundError:
            print('PODTensors.csv or PODEigenValues.csv not found. Continuing with POD tensor coefficients set to 0')
            PODTensors = np.zeros((len(PODArray), 9), dtype=complex)
            PODEigenValues = np.zeros((len(PODArray), 3), dtype=complex)

        cutoff = u1Truncated.shape[1]
        print(' Loaded Data')

    # save_U = True
    if save_U is True and recoverymode is False:
        np.save('Results/' + sweepname + '/Data/U1_truncated', u1Truncated)
        np.save('Results/' + sweepname + '/Data/U2_truncated', u2Truncated)
        np.save('Results/' + sweepname + '/Data/U3_truncated', u3Truncated)
        np.savetxt('Results/' + sweepname + '/Data/PODTensors.csv', PODTensors, delimiter=',')
        np.savetxt('Results/' + sweepname + '/Data/PODEigenvalues.csv', PODEigenValues, delimiter=',')
    ########################################################################
    # Create the ROM

    print(' creating reduced order model', end='\r')
    # Mu0=4*np.pi*10**(-7)
    nu_no_omega = Mu0 * (alpha ** 2)

    Theta_0 = GridFunction(fes)
    u, v = fes2.TnT()

    if BigProblem == True:
        a0 = BilinearForm(fes2, symmetric=True, bonus_intorder=Additional_Int_Order)
    else:
        a0 = BilinearForm(fes2, symmetric=True)
    a0 += SymbolicBFI((mu ** (-1)) * InnerProduct(curl(u), curl(v)), bonus_intorder=Additional_Int_Order)
    a0 += SymbolicBFI((1j) * (1 - inout) * epsi * InnerProduct(u, v), bonus_intorder=Additional_Int_Order)
    if BigProblem == True:
        a1 = BilinearForm(fes2, symmetric=True)
    else:
        a1 = BilinearForm(fes2, symmetric=True)
    a1 += SymbolicBFI((1j) * inout * nu_no_omega * sigma * InnerProduct(u, v), bonus_intorder=Additional_Int_Order)

    a0.Assemble()
    a1.Assemble()

    Theta_0.vec.FV().NumPy()[:] = Theta0Sol[:, 0]
    r1 = LinearForm(fes2)
    r1 += SymbolicLFI(inout * (-1j) * nu_no_omega * sigma * InnerProduct(Theta_0, v), bonus_intorder=Additional_Int_Order)
    r1 += SymbolicLFI(inout * (-1j) * nu_no_omega * sigma * InnerProduct(xivec[0], v), bonus_intorder=Additional_Int_Order)
    r1.Assemble()
    read_vec = r1.vec.CreateVector()
    write_vec = r1.vec.CreateVector()

    Theta_0.vec.FV().NumPy()[:] = Theta0Sol[:, 1]
    r2 = LinearForm(fes2)
    r2 += SymbolicLFI(inout * (-1j) * nu_no_omega * sigma * InnerProduct(Theta_0, v), bonus_intorder=Additional_Int_Order)
    r2 += SymbolicLFI(inout * (-1j) * nu_no_omega * sigma * InnerProduct(xivec[1], v), bonus_intorder=Additional_Int_Order)
    r2.Assemble()

    Theta_0.vec.FV().NumPy()[:] = Theta0Sol[:, 2]
    r3 = LinearForm(fes2)
    r3 += SymbolicLFI(inout * (-1j) * nu_no_omega * sigma * InnerProduct(Theta_0, v), bonus_intorder=Additional_Int_Order)
    r3 += SymbolicLFI(inout * (-1j) * nu_no_omega * sigma * InnerProduct(xivec[2], v), bonus_intorder=Additional_Int_Order)
    r3.Assemble()

    if PODErrorBars == True:
        fes0 = HCurl(mesh, order=0, dirichlet="outer", complex=True, gradientdomains=dom_nrs_metal)
        ndof0 = fes0.ndof
        RerrorReduced1 = np.zeros([ndof0, cutoff * 2 + 1], dtype=complex)
        RerrorReduced2 = np.zeros([ndof0, cutoff * 2 + 1], dtype=complex)
        RerrorReduced3 = np.zeros([ndof0, cutoff * 2 + 1], dtype=complex)
        ProH = GridFunction(fes2)
        ProL = GridFunction(fes0)
    ########################################################################
    # Create the ROM
    R1 = r1.vec.FV().NumPy()
    R2 = r2.vec.FV().NumPy()
    R3 = r3.vec.FV().NumPy()
    A0H = np.zeros([ndof2, cutoff], dtype=complex)
    A1H = np.zeros([ndof2, cutoff], dtype=complex)

    # E1
    for i in range(cutoff):
        read_vec.FV().NumPy()[:] = u1Truncated[:, i]
        write_vec.data = a0.mat * read_vec
        A0H[:, i] = write_vec.FV().NumPy()
        write_vec.data = a1.mat * read_vec
        A1H[:, i] = write_vec.FV().NumPy()
    HA0H1 = (np.conjugate(np.transpose(u1Truncated)) @ A0H)
    HA1H1 = (np.conjugate(np.transpose(u1Truncated)) @ A1H)
    HR1 = (np.conjugate(np.transpose(u1Truncated)) @ np.transpose(R1))

    if PODErrorBars == True:
        ProH.vec.FV().NumPy()[:] = R1
        ProL.Set(ProH)
        RerrorReduced1[:, 0] = ProL.vec.FV().NumPy()[:]
        for i in range(cutoff):
            ProH.vec.FV().NumPy()[:] = A0H[:, i]
            ProL.Set(ProH)
            RerrorReduced1[:, i + 1] = ProL.vec.FV().NumPy()[:]
            ProH.vec.FV().NumPy()[:] = A1H[:, i]
            ProL.Set(ProH)
            RerrorReduced1[:, i + cutoff + 1] = ProL.vec.FV().NumPy()[:]
    # E2
    for i in range(cutoff):
        read_vec.FV().NumPy()[:] = u2Truncated[:, i]
        write_vec.data = a0.mat * read_vec
        A0H[:, i] = write_vec.FV().NumPy()
        write_vec.data = a1.mat * read_vec
        A1H[:, i] = write_vec.FV().NumPy()
    HA0H2 = (np.conjugate(np.transpose(u2Truncated)) @ A0H)
    HA1H2 = (np.conjugate(np.transpose(u2Truncated)) @ A1H)
    HR2 = (np.conjugate(np.transpose(u2Truncated)) @ np.transpose(R2))

    if PODErrorBars == True:
        ProH.vec.FV().NumPy()[:] = R2
        ProL.Set(ProH)
        RerrorReduced2[:, 0] = ProL.vec.FV().NumPy()[:]
        for i in range(cutoff):
            ProH.vec.FV().NumPy()[:] = A0H[:, i]
            ProL.Set(ProH)
            RerrorReduced2[:, i + 1] = ProL.vec.FV().NumPy()[:]
            ProH.vec.FV().NumPy()[:] = A1H[:, i]
            ProL.Set(ProH)
            RerrorReduced2[:, i + cutoff + 1] = ProL.vec.FV().NumPy()[:]
    # E3
    for i in range(cutoff):
        read_vec.FV().NumPy()[:] = u3Truncated[:, i]
        write_vec.data = a0.mat * read_vec
        A0H[:, i] = write_vec.FV().NumPy()
        write_vec.data = a1.mat * read_vec
        A1H[:, i] = write_vec.FV().NumPy()
    HA0H3 = (np.conjugate(np.transpose(u3Truncated)) @ A0H)
    HA1H3 = (np.conjugate(np.transpose(u3Truncated)) @ A1H)
    HR3 = (np.conjugate(np.transpose(u3Truncated)) @ np.transpose(R3))

    if PODErrorBars == True:
        ProH.vec.FV().NumPy()[:] = R3
        ProL.Set(ProH)
        RerrorReduced3[:, 0] = ProL.vec.FV().NumPy()[:]
        for i in range(cutoff):
            ProH.vec.FV().NumPy()[:] = A0H[:, i]
            ProL.Set(ProH)
            RerrorReduced3[:, i + 1] = ProL.vec.FV().NumPy()[:]
            ProH.vec.FV().NumPy()[:] = A1H[:, i]
            ProL.Set(ProH)
            RerrorReduced3[:, i + cutoff + 1] = ProL.vec.FV().NumPy()[:]

    # Clear the variables
    A0H, A1H = None, None
    a0, a1 = None, None

    ########################################################################
    # Sort out the error bounds
    if PODErrorBars == True:
        if BigProblem == True:
            MR1 = np.zeros([ndof0, cutoff * 2 + 1], dtype=np.complex64)
            MR2 = np.zeros([ndof0, cutoff * 2 + 1], dtype=np.complex64)
            MR3 = np.zeros([ndof0, cutoff * 2 + 1], dtype=np.complex64)
        else:
            MR1 = np.zeros([ndof0, cutoff * 2 + 1], dtype=complex)
            MR2 = np.zeros([ndof0, cutoff * 2 + 1], dtype=complex)
            MR3 = np.zeros([ndof0, cutoff * 2 + 1], dtype=complex)

        u, v = fes0.TnT()

        m = BilinearForm(fes0)
        m += SymbolicBFI(InnerProduct(u, v), bonus_intorder=Additional_Int_Order)
        f = LinearForm(fes0)
        m.Assemble()
        c = Preconditioner(m, "local")
        c.Update()
        inverse = CGSolver(m.mat, c.mat, precision=1e-20, maxsteps=500)

        ErrorGFU = GridFunction(fes0)
        for i in range(2 * cutoff + 1):
            # E1
            ProL.vec.data.FV().NumPy()[:] = RerrorReduced1[:, i]
            ProL.vec.data -= m.mat * ErrorGFU.vec
            ErrorGFU.vec.data += inverse * ProL.vec
            MR1[:, i] = ErrorGFU.vec.FV().NumPy()

            # E2
            ProL.vec.data.FV().NumPy()[:] = RerrorReduced2[:, i]
            ProL.vec.data -= m.mat * ErrorGFU.vec
            ErrorGFU.vec.data += inverse * ProL.vec
            MR2[:, i] = ErrorGFU.vec.FV().NumPy()

            # E3
            ProL.vec.data.FV().NumPy()[:] = RerrorReduced3[:, i]
            ProL.vec.data -= m.mat * ErrorGFU.vec
            ErrorGFU.vec.data += inverse * ProL.vec
            MR3[:, i] = ErrorGFU.vec.FV().NumPy()

        G_Store = np.zeros([2 * cutoff + 1, 2 * cutoff + 1, 6], dtype=complex)
        G_Store[:, :, 0] = np.transpose(np.conjugate(RerrorReduced1)) @ MR1
        G_Store[:, :, 1] = np.transpose(np.conjugate(RerrorReduced2)) @ MR2
        G_Store[:, :, 2] = np.transpose(np.conjugate(RerrorReduced3)) @ MR3
        G_Store[:, :, 3] = np.transpose(np.conjugate(RerrorReduced1)) @ MR2
        G_Store[:, :, 4] = np.transpose(np.conjugate(RerrorReduced1)) @ MR3
        G_Store[:, :, 5] = np.transpose(np.conjugate(RerrorReduced2)) @ MR3

        # Clear the variables
        RerrorReduced1, RerrorReduced2, RerrorReduced3 = None, None, None
        MR1, MR2, MR3 = None, None, None
        fes0, m, c, inverse = None, None, None, None

        fes3 = HCurl(mesh, order=Order, dirichlet="outer", gradientdomains=dom_nrs_metal)
        ndof3 = fes3.ndof
        Omega = Array[0]
        u, v = fes3.TnT()
        amax = BilinearForm(fes3)
        amax += SymbolicBFI((mu ** (-1)) * curl(u) * curl(v), bonus_intorder=Additional_Int_Order)
        amax += SymbolicBFI((1 - inout) * epsi * u * v, bonus_intorder=Additional_Int_Order)
        amax += SymbolicBFI(inout * sigma * (alpha ** 2) * Mu0 * Omega * u * v, bonus_intorder=Additional_Int_Order)

        m = BilinearForm(fes3)
        m += SymbolicBFI(u * v, bonus_intorder=Additional_Int_Order)

        apre = BilinearForm(fes3)
        apre += SymbolicBFI(curl(u) * curl(v), bonus_intorder=Additional_Int_Order)
        apre += SymbolicBFI(u * v, bonus_intorder=Additional_Int_Order)
        pre = Preconditioner(amax, "bddc")

        with TaskManager():
            amax.Assemble()
            m.Assemble()
            apre.Assemble()

            # build gradient matrix as sparse matrix (and corresponding scalar FESpace)
            gradmat, fesh1 = fes3.CreateGradient()
            gradmattrans = gradmat.CreateTranspose()  # transpose sparse matrix
            math1 = gradmattrans @ m.mat @ gradmat  # multiply matrices
            math1[0, 0] += 1  # fix the 1-dim kernel
            invh1 = math1.Inverse(inverse="sparsecholesky")

            # build the Poisson projector with operator Algebra:
            proj = IdentityMatrix() - gradmat @ invh1 @ gradmattrans @ m.mat
            projpre = proj @ pre.mat
            evals, evecs = solvers.PINVIT(amax.mat, m.mat, pre=projpre, num=1, maxit=50, printrates=False)

        alphaLB = evals[0]
        # print(f'alphaLB = {alphaLB}')

    else:
        alphaLB, G_Store = False, False

        # Clear the variables
        fes3, amax, apre, pre, invh1, m = None, None, None, None, None, None
    timing_dictionary['ROM'] = time.time()

    ######################################################################
    # Produce the sweep on the lower dimensional space
    g = np.zeros([cutoff, NumberofFrequencies, 3], dtype=complex)
    for k, omega in enumerate(Array):
        g[:, k, 0] = np.linalg.solve(HA0H1 + HA1H1 * omega, HR1 * omega)
        g[:, k, 1] = np.linalg.solve(HA0H2 + HA1H2 * omega, HR2 * omega)
        g[:, k, 2] = np.linalg.solve(HA0H3 + HA1H3 * omega, HR3 * omega)
    # Work out where to send each frequency
    timing_dictionary['SolvedSmallerSystem'] = time.time()
    Tensor_CPUs = min(NumberofFrequencies, multiprocessing.cpu_count(), CPUs)
    Tensor_CPUs = 1

    Core_Distribution = []
    Count_Distribution = []
    for i in range(Tensor_CPUs):
        Core_Distribution.append([])
        Count_Distribution.append([])
    # Distribute frequencies between the cores
    CoreNumber = 0
    for i, Omega in enumerate(Array):
        Core_Distribution[CoreNumber].append(Omega)
        Count_Distribution[CoreNumber].append(i)
        if CoreNumber == Tensor_CPUs - 1:
            CoreNumber = 0
        else:
            CoreNumber += 1
    # Distribute the lower dimensional solutions
    Lower_Sols = []
    for i in range(Tensor_CPUs):
        TempArray = np.zeros([cutoff, len(Count_Distribution[i]), 3], dtype=complex)
        for j, Sim in enumerate(Count_Distribution[i]):
            TempArray[:, j, :] = g[:, Sim, :]
        Lower_Sols.append(TempArray)

    timing_dictionary['AssignedCores'] = time.time()


    # Depending on if the user has specified using the slower integral method. This is known to produce the correct
    # answer. Also used if PODErrorBars are required, since it calculates error certificates at the same time as the
    # tensor coefficients.
    use_integral_debug = False
    if use_integral is True or use_integral_debug is True:
        # Cteate the inputs
        Runlist = []
        manager = multiprocessing.Manager()
        counter = manager.Value('i', 0)
        for i in range(Tensor_CPUs):
            Runlist.append((Core_Distribution[i], mesh, fes, fes2, Lower_Sols[i], u1Truncated, u2Truncated, u3Truncated,
                            Theta0Sol, xivec, alpha, sigma, mu, inout, N0, NumberofFrequencies, counter, PODErrorBars,
                            alphaLB, G_Store, Order, Integration_Order, Additional_Int_Order, use_integral))

        # Run on the multiple cores
        # Edit James Elgy: changed how pool was generated to 'spawn': see
        # https://britishgeologicalsurvey.github.io/science/python-forking-vs-spawn/
        with multiprocessing.get_context('spawn').Pool(Tensor_CPUs) as pool:
            Outputs = pool.starmap(Theta1_Lower_Sweep, Runlist)

    else:

        print(Integration_Order)
        print(Additional_Int_Order)
        print(Additional_Int_Order)
        print(f'{Integration_Order - 2*(Order+1)}')


        # Constructing 𝐊ᵢⱼ (eqn 7 from paper)
        # For the K bilinear forms, and also later bilinear and linear forms, we specify an integration order specific
        # to the postprocessing. See comment in main.py on the topic.
        u, v = fes2.TnT()
        K = BilinearForm(fes2, symmetric=True)
        K += SymbolicBFI(inout * mu ** (-1) * curl(u) * Conj(curl(v)), bonus_intorder=Integration_Order - 2*(Order+1))
        K += SymbolicBFI((1 - inout) * curl(u) * Conj(curl(v)), bonus_intorder=Integration_Order - 2*(Order+1))
        K.Assemble()
        rows, cols, vals = K.mat.COO()
        del K
        Q = sp.csr_matrix((vals, (rows, cols)))
        del rows, cols, vals
        gc.collect()

        # For faster computation of tensor coefficients, we multiply with Ui before the loop.
        # This computes MxM 𝐊ᴹᵢⱼ. For each of the combinations ij we store the smaller matrix rather than recompute in
        # each case.
        Q11 = np.conj(np.transpose(u1Truncated)) @ Q @ u1Truncated
        Q22 = np.conj(np.transpose(u2Truncated)) @ Q @ u2Truncated
        Q33 = np.conj(np.transpose(u3Truncated)) @ Q @ u3Truncated
        Q21 = np.conj(np.transpose(u2Truncated)) @ Q @ u1Truncated
        Q31 = np.conj(np.transpose(u3Truncated)) @ Q @ u1Truncated
        Q32 = np.conj(np.transpose(u3Truncated)) @ Q @ u2Truncated

        del Q
        Q_array = [Q11, Q22, Q33, Q21, Q31, Q32]


        # Similar for 𝐂ᴹᵢⱼ. refered to as A in code. For each of the combinations ij we store the smaller matrix rather
        # than recompute in each case.
        # Using the same basis functions for both the theta0 and theta1 problems allows us to reduce the number of
        # bilinear forms that need to be constructed.
        # For 𝐍ᴷ = (𝐍₀)ᴷ then 𝐂 = 𝐂¹ = 𝐂² and 𝐬ᵢ = 𝐭ᵢ. In this way we only need to consider 𝐂 (called A in code), 𝐬
        # (called E in code) and c (called G in code) from paper.
        A = BilinearForm(fes2, symmetric=True)
        A += SymbolicBFI(sigma * inout * (v * u), bonus_intorder=Integration_Order - 2*(Order+1))
        A.Assemble()
        rows, cols, vals = A.mat.COO()
        del A
        A_mat = sp.csr_matrix((vals, (rows, cols)))

        del rows, cols, vals
        gc.collect()

        E = np.zeros((3, fes2.ndof), dtype=complex)
        G = np.zeros((3, 3))

        for i in range(3):

            E_lf = LinearForm(fes2)
            E_lf += SymbolicLFI(sigma * inout * xivec[i] * v, bonus_intorder=Integration_Order - 2*(Order+1))
            E_lf.Assemble()
            E[i, :] = E_lf.vec.FV().NumPy()[:]
            del E_lf

            for j in range(3):
                G[i, j] = Integrate(sigma * inout * xivec[i] * xivec[j], mesh, order=Integration_Order)

        H = E.transpose()

        print(' Built K, Q, E, and G')

        #Testing:
        # run_test_comparison(u,v, sigma, xivec, inout, mesh, Theta0Sol, Lower_Sols, u1Truncated, fes, fes2)

        # Similarly for the imaginary part, we multiply with the theta0 sols beforehand.
        A_mat_t0_1 = (A_mat) @ Theta0Sol[:, 0]
        A_mat_t0_2 = (A_mat) @ Theta0Sol[:, 1]
        A_mat_t0_3 = (A_mat) @ Theta0Sol[:, 2]


        # (𝐂)^M being the reduced MxM complex matrix. Similarly to the real part, we store each combination of i,j.
        T11 = np.conj(np.transpose(u1Truncated)) @ A_mat @ u1Truncated
        T22 = np.conj(np.transpose(u2Truncated)) @ A_mat @ u2Truncated
        T33 = np.conj(np.transpose(u3Truncated)) @ A_mat @ u3Truncated
        T21 = np.conj(np.transpose(u2Truncated)) @ A_mat @ u1Truncated
        T31 = np.conj(np.transpose(u3Truncated)) @ A_mat @ u1Truncated
        T32 = np.conj(np.transpose(u3Truncated)) @ A_mat @ u2Truncated

        T_array = [T11, T22, T33, T21, T31, T32]

        # At this point, we have constructed each of the main matrices we need and obtained the reduced A matrix. The
        # larger bilinear form can therefore be removed to save memory.
        del A_mat

        At0_array = [A_mat_t0_1, A_mat_t0_2, A_mat_t0_3]

        # Here we compute (𝐨ⱼ)ᵀ (̅𝐂²)ᴹ
        # Renamed to better fit naming convention
        UAt011_conj = np.conj(u1Truncated.transpose()) @ A_mat_t0_1
        UAt022_conj = np.conj(u2Truncated.transpose()) @ A_mat_t0_2
        UAt033_conj = np.conj(u3Truncated.transpose()) @ A_mat_t0_3
        UAt012_conj = np.conj(u1Truncated.transpose()) @ A_mat_t0_2
        UAt013_conj = np.conj(u1Truncated.transpose()) @ A_mat_t0_3
        UAt023_conj = np.conj(u2Truncated.transpose()) @ A_mat_t0_3

        UAt0_conj = [UAt011_conj, UAt022_conj, UAt033_conj, UAt012_conj, UAt013_conj, UAt023_conj]


        # Similarly we compute and store (𝐨ⱼ)ᵀ (𝐂²)ᴹ
        UAt011 = (u1Truncated.transpose()) @ A_mat_t0_1
        UAt022 = (u2Truncated.transpose()) @ A_mat_t0_2
        UAt033 = (u3Truncated.transpose()) @ A_mat_t0_3
        UAt021 = (u2Truncated.transpose()) @ A_mat_t0_1
        UAt031 = (u3Truncated.transpose()) @ A_mat_t0_1
        UAt032 = (u3Truncated.transpose()) @ A_mat_t0_2
        UAt0U_array = [UAt011, UAt022, UAt033, UAt021, UAt031, UAt032]


        # Finally, we can construct constants that do not depend on frequency.
        # the constant c1 corresponds to 𝐨ⱼᵀ 𝐂⁽¹⁾ 𝐨ᵢ. Similar to other cases we store each combination of i and j.
        c1_11 = (np.transpose(Theta0Sol[:, 0])) @ A_mat_t0_1
        c1_22 = (np.transpose(Theta0Sol[:, 1])) @ A_mat_t0_2
        c1_33 = (np.transpose(Theta0Sol[:, 2])) @ A_mat_t0_3
        c1_21 = (np.transpose(Theta0Sol[:, 1])) @ A_mat_t0_1
        c1_31 = (np.transpose(Theta0Sol[:, 2])) @ A_mat_t0_1
        c1_32 = (np.transpose(Theta0Sol[:, 2])) @ A_mat_t0_2

        # c5 corresponds to 𝐬ᵢᵀ 𝐨ⱼ. Note that E has been transposed here.
        c5_11 = E[0, :] @ Theta0Sol[:, 0]
        c5_22 = E[1, :] @ Theta0Sol[:, 1]
        c5_33 = E[2, :] @ Theta0Sol[:, 2]
        c5_21 = E[1, :] @ Theta0Sol[:, 0]
        c5_31 = E[2, :] @ Theta0Sol[:, 0]
        c5_32 = E[2, :] @ Theta0Sol[:, 1]

        # Similarly to other examples we store each combination rather than recompute
        c1_array = [c1_11, c1_22, c1_33, c1_21, c1_31, c1_32]
        c5_array = [c5_11, c5_22, c5_33, c5_21, c5_31, c5_32]

        # c7 = G corresponds to cᵢⱼ from paper. Note that G does not depend on the FEM basis functions, rather is a
        # polynomial.
        c7 = G

        # c8 corresponds to  𝐬ⱼᵀ 𝐨ᵢ and shold equal c5 for on diagonal entries.
        c8_11 = Theta0Sol[:, 0] @ H[:, 0]
        c8_22 = Theta0Sol[:, 1] @ H[:, 1]
        c8_33 = Theta0Sol[:, 2] @ H[:, 2]
        c8_21 = Theta0Sol[:, 1] @ H[:, 0]
        c8_31 = Theta0Sol[:, 2] @ H[:, 0]
        c8_32 = Theta0Sol[:, 2] @ H[:, 1]

        c8_array = [c8_11, c8_22, c8_33, c8_21, c8_31, c8_32]

        # EU is the reduced linear form for E. Here we compute (̅𝐭ᴹ)ᵀ.
        EU_11 = E[0, :] @ np.conj(u1Truncated)
        EU_22 = E[1, :] @ np.conj(u2Truncated)
        EU_33 = E[2, :] @ np.conj(u3Truncated)
        EU_21 = E[1, :] @ np.conj(u1Truncated)
        EU_31 = E[2, :] @ np.conj(u1Truncated)
        EU_32 = E[2, :] @ np.conj(u2Truncated)

        EU_array_conj = [EU_11, EU_22, EU_33, EU_21, EU_31, EU_32]

        H = E.transpose()

        # also computing  (𝐭ᴹ)ᵀ
        # Renamed to better fit naming convention
        UH_11 = u1Truncated.transpose() @ H[:, 0]
        UH_22 = u2Truncated.transpose() @ H[:, 1]
        UH_33 = u3Truncated.transpose() @ H[:, 2]
        UH_21 = u2Truncated.transpose() @ H[:, 0]
        UH_31 = u3Truncated.transpose() @ H[:, 0]
        UH_32 = u3Truncated.transpose() @ H[:, 1]

        UH_array = [UH_11, UH_22, UH_33, UH_21, UH_31, UH_32]

        timing_dictionary['BuildSystemMatrices'] = time.time()

        runlist = []
        for i in range(Tensor_CPUs):
            runlist.append((Core_Distribution[i], Q_array, c1_array, c5_array, c7, c8_array, At0_array, UAt0_conj,
                            UAt0U_array, T_array, EU_array_conj, UH_array, Lower_Sols[i], G_Store, cutoff, fes2.ndof,
                            alpha, False))

        with multiprocessing.get_context('spawn').Pool(Tensor_CPUs) as pool:
            Outputs = pool.starmap(Theta1_Lower_Sweep_Mat_Method, runlist)

    try:
        pool.terminate()
        print('manually closed pool')
    except:
        print('Pool has already closed.')



    # Unpack the outputs
    if use_integral is True or use_integral_debug is True:
        if PODErrorBars == True:
            ErrorTensors = np.zeros([NumberofFrequencies, 6])
        for i, Output in enumerate(Outputs):
            for j, Num in enumerate(Count_Distribution[i]):
                if PODErrorBars == True:
                    TensorArray[Num, :] = Output[0][j]
                    EigenValues[Num, :] = Output[1][j]
                    ErrorTensors[Num, :] = Output[2][j]
                else:
                    TensorArray[Num, :] = Output[0][j]
                    EigenValues[Num, :] = Output[1][j]

    else:
        for i, Output in enumerate(Outputs):
            for j, Num in enumerate(Count_Distribution[i]):
                if PODErrorBars == True:
                    TensorArray[Num, :] = Output[0][j]
                    TensorArray[Num, :] = Output[0][j] + N0.flatten()
                    R = TensorArray[Num, :].real.reshape(3, 3)
                    I = TensorArray[Num, :].imag.reshape(3, 3)
                    EigenValues[Num, :] = np.sort(np.linalg.eigvals(R)) + 1j * np.sort(np.linalg.eigvals(I))
                    # ErrorTensors[Num, :] = Output[2][j]
                else:
                    TensorArray[Num, :] = Output[0][j] + N0.flatten()
                    R = TensorArray[Num, :].real.reshape(3, 3)
                    I = TensorArray[Num, :].imag.reshape(3, 3)
                    EigenValues[Num, :] = np.sort(np.linalg.eigvals(R)) + 1j * np.sort(np.linalg.eigvals(I))

    print(' reduced order systems solved')

    ### EXPORTING TO VTK
    # sweepname = 'Results/'
    # dom_nrs_metal = [0 if mat == 'air' else 1 for mat in mesh.GetMaterials()]
    # for index, om in enumerate(Array):
    #     # Projecting POD solutions to higher dimensional space.
    #     W1 = np.dot(u1Truncated, Lower_Sols[:, index, 0]).flatten()
    #     W2 = np.dot(u2Truncated, Lower_Sols[:, index, 1]).flatten()
    #     W3 = np.dot(u2Truncated, Lower_Sols[:, index, 3]).flatten()
    #
    #     generate_VTK(mesh, fes2, [W1, W2, W3], om, sigma, sweepname, alpha, dom_nrs_metal)


    if (use_integral is False) and (use_integral_debug is False) and (PODErrorBars is True):
        print(' Computing Errors')
        # For parallelisation, this has to be a separate function. Also with the intention that we can also reuse this
        # function in the other POD functions. I.e. PODSweep, PODSweepMulti.
        ErrorTensors = np.zeros((len(Array), 6))
        for i in range(Tensor_CPUs):
            Distributed_Errors = calc_error_certificates(Core_Distribution[i], alphaLB, G_Store, cutoff, alpha, Lower_Sols[i])
            ErrorTensors[Count_Distribution[i],:] = Distributed_Errors

    print(' frequency sweep complete')
    timing_dictionary['Tensors'] = time.time()
    np.save('Results/' + sweepname + f'/Data/Timings_cpus={CPUs}.npy', timing_dictionary)

    if PlotPod == True:
        if PODErrorBars == True:
            return TensorArray, EigenValues, N0, PODTensors, PODEigenValues, numelements, ErrorTensors, (ndof, ndof2)
        else:
            return TensorArray, EigenValues, N0, PODTensors, PODEigenValues, numelements, (ndof, ndof2)
    else:
        if PODErrorBars == True:
            return TensorArray, EigenValues, N0, numelements, ErrorTensors, (ndof, ndof2)
        else:
            return TensorArray, EigenValues, N0, numelements, (ndof, ndof2)


