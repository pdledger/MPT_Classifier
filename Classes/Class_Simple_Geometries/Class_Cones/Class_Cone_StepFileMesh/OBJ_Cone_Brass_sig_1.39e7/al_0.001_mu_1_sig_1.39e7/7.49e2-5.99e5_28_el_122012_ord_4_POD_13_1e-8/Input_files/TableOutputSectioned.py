#This code takes the inputs from main.py and Settings.py and produces the lines for a tex file that will produce a table of inputs
#Importing
import numpy as np
from ngsolve import *

#extract the information from main.py
f = open("main.py","r")
f1 = f.readlines()
for line in f1:
    if line.find("#Main script") == -1:
        exec(line)
    else:
        break
f.close()

#extract the information from Settings.py
from Settings import *
CPUs,BigProblem,PODPoints,PODTol,OldMesh = DefaultSettings()
PlotPod, PODErrorBars, EddyCurrentTest, vtk_output, Refine_vtk = AdditionalOutputs()
FolderName = SaverSettings()
Solver,epsi,Maxsteps,Tolerance = SolverParameters()
        


Lines = ["\\begin{table}[H]\n"]
Lines.append("\\begin{center}\n")
Lines.append("\\large{\\texttt{main.py}}\\normalsize{ }\\\\\\vspace{0.2cm}\n")
Lines.append("\\begin{tabular}{!\\vrule p{4.5cm}!\\vrule p{4.5cm}!\\vrule p{4.5cm}!\\vrule}\n")
Lines.append("\\hline\n")
Lines.append('\\texttt{Geometry = "'+Geometry+'"} & \\texttt{alpha = '+str(alpha)+"} & \\texttt{MeshSize = "+str(MeshSize)+"}\\\\\\hline\n")
Lines.append("\\texttt{Order = "+str(Order)+"} & \\texttt{Start = "+str(Start)+"} & \\texttt{Finish = "+str(Finish)+"}\\\\\\hline\n")
Lines.append("\\texttt{Points = "+str(Points)+"} & \\texttt{Single = "+str(Single)+"} &\\texttt{Omega = "+str(Omega)+"}\\\\\\hline\n")
Lines.append("\\end{tabular}\\\\\n")
Lines.append("\\begin{tabular}{!\\vrule p{4.5cm}!\\vrule p{4.5cm}!\\vrule}\n")
Lines.append("\\texttt{Pod = "+str(Pod)+"} & \\texttt{MultiProcessing = "+str(MultiProcessing)+"}\\\\\\hline\n")
Lines.append("\\end{tabular}\n")
Lines.append("\\\\\\vspace{0.5cm}\\large{\\texttt{Settings.py}}\\normalsize{ }\\\\\\vspace{0.2cm}\n")
Lines.append("\\begin{tabular}{!\\vrule p{4.5cm}!\\vrule p{4.5cm}!\\vrule p{4.5cm}!\\vrule}\n")
Lines.append("\\hline\n")
Lines.append("\\texttt{CPUs = "+str(CPUs)+"} & \\texttt{BigProblem = "+str(BigProblem)+"} & \\texttt{PODPoints = "+str(PODPoints)+"}\\\\\\hline\n")
Lines.append("\\texttt{PODTol = "+str(PODTol)+"} & \\texttt{OldMesh = "+str(OldMesh)+"} & \\texttt{PlotPod = "+str(PlotPod)+"}\\\\\\hline\n")
Lines.append("\\texttt{PODErrorBars = "+str(PODErrorBars)+"} & \\texttt{EddyCurrentTest = "+str(EddyCurrentTest)+"} & \\texttt{vtk\\_output = "+str(vtk_output)+"}\\\\\\hline\n")
Lines.append("\\texttt{Refine\\_vtk = "+str(Refine_vtk)+'} & \\texttt{FolderName = "'+FolderName+'"} & \\texttt{Solver = "'+Solver+'"}\\\\\\hline\n')
Lines.append("\\texttt{epsi = "+str(epsi)+"} & \\texttt{Maxsteps = "+str(Maxsteps)+"} & \\texttt{Tolerance = "+str(Tolerance)+"}\\\\\\hline\n")
Lines.append("\\end{tabular}\\\\\n")
Lines.append("\\begin{tabular}{!\\vrule p{4.5cm}!\\vrule}\n")
Lines.append("\\texttt{ngsglobals.msg\\_level = "+str(ngsglobals.msg_level)+"}\\\\\\hline\n")
Lines.append("\\end{tabular}\n")
Lines.append("\\caption{Add a caption.}\\label{Add a label}\n")
Lines.append("\\end{center}\n")
Lines.append("\\end{table}\n")








f = open("Output.txt","w+")
for line in Lines:
    f.write(line)
f.close()