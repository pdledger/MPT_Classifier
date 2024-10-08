#Importing
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from .TickFormatter import *

#This is required for when a copy of the file is sent to the results section
try:
    sys.path.insert(0,"Settings")
except:
    pass

from PlotterSettings import PlotterSettings


def TensorPlotter(savename, Array, Values, EddyCurrentTest):
    """_summary_
    B.A. Wilson, J.Elgy, P.D. Ledger.
    Function to plot tensor coefficients.

    Args:
        savename (str): path to save figure.
        Array (list): frequency array
        Values (np.ndarray): Nx6 complex tensor coefficients
        EddyCurrentTest (float | None): if using eddy current test, max frequency, else None

    Returns:
        bool: plot figure or not.
    """
    
    
    # Create a way to reference xkcd colours
    PYCOL = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
             '#17becf']

    # Retrieve the settings for the plot
    Title, Show, _, TTP, MLS, MMS, _, _, _, _, ECL = PlotterSettings()

    # Plot the graph
    fig, ax = plt.subplots()

    # Plot the mainlines
    for i, line in enumerate(TTP):
        if i == 0:
            lines = ax.plot(Array, Values[:, line - 1].real, MLS, markersize=MMS, color=PYCOL[i])
        else:
            lines += ax.plot(Array, Values[:, line - 1].real, MLS, markersize=MMS, color=PYCOL[i])

    ymin, ymax = ax.get_ylim()

    # Format the axes
    plt.xscale('log')
    plt.ylim(ymin, ymax)
    ax.grid(True)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(TickFormatter))
    plt.subplots_adjust(wspace=0.6, hspace=0.6, left=0.15, bottom=0.1, right=0.94, top=0.90)

    # Label the axes
    plt.xlabel("Frequency (rad/s)")
    plt.ylabel(r"$\mathcal{N}^0_{ij}+\mathcal{R}_{ij}$")

    if Title == True:
        plt.title(r"Tensor coefficients of $\mathcal{N}^0+\mathcal{R}$")

    # Create the legend
    names = []
    CoefficientRef = ["11", "12", "13", "22", "23", "33", "21", "31", "_", "32"]
    for i, number in enumerate(TTP):
        if number == 1 or number == 4 or number == 6:
            names.append(r"Re($\mathcal{M}_{" + CoefficientRef[number - 1] + "}(\omega)$)")
        else:
            names.append(
                r"Re($\mathcal{M}_{" + CoefficientRef[number - 1] + "}(\omega)$)=Re($\mathcal{M}_{" + CoefficientRef[
                    number + 4] + "}(\omega)$)")

    # Show where the eddy-current breaks down (if applicable)
    if isinstance(EddyCurrentTest, float):
        if ECL == True:
            x = np.ones(10) * EddyCurrentTest
            y = np.linspace(ymin, ymax, 10)
            lines += ax.plot(x, y, '--r')
            names.append(r"eddy-current model valid")

    # Make the legend
    ax.legend(lines, names)

    # Save the graph
    plt.savefig(savename + "RealTensorCoeficients.pdf")

    # Plot the imaginary graph
    fig, ax = plt.subplots()

    # Plot the mainlines
    for i, line in enumerate(TTP):
        if i == 0:
            lines = ax.plot(Array, Values[:, line - 1].imag, MLS, markersize=MMS, color=PYCOL[i])
        else:
            lines += ax.plot(Array, Values[:, line - 1].imag, MLS, markersize=MMS, color=PYCOL[i])

    ymin, ymax = ax.get_ylim()

    # Format the axes
    plt.xscale('log')
    plt.ylim(ymin, ymax)
    ax.grid(True)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(TickFormatter))
    plt.subplots_adjust(wspace=0.6, hspace=0.6, left=0.15, bottom=0.1, right=0.94, top=0.90)

    # Label the axes
    plt.xlabel("Frequency (rad/s)")
    plt.ylabel(r"$\mathcal{I}_{ij}$")

    if Title == True:
        plt.title(r"Tensor coefficients of $\mathcal{I}$")

    # Create the legend
    names = []
    for i, number in enumerate(TTP):
        if number == 1 or number == 4 or number == 6:
            names.append(r"Im($\mathcal{M}_{" + CoefficientRef[number - 1] + "}(\omega)$)")
        else:
            names.append(
                r"Im($\mathcal{M}_{" + CoefficientRef[number - 1] + "}(\omega)$)=Im($\mathcal{M}_{" + CoefficientRef[
                    number + 4] + "}(\omega)$)")

    # Show where the eddy-current breaks down (if applicable)
    if isinstance(EddyCurrentTest, float):
        if ECL == True:
            x = np.ones(10) * EddyCurrentTest
            y = np.linspace(ymin, ymax, 10)
            lines += ax.plot(x, y, '--r')
            names.append(r"eddy-current model valid")

    # Make the legend
    ax.legend(lines, names)

    # Save the graph
    plt.savefig(savename + "ImaginaryTensorCoeficients.pdf")

    return Show


