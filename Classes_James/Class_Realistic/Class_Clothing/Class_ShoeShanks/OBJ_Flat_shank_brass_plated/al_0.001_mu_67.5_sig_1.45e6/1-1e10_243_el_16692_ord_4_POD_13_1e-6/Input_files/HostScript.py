import numpy as np
import os
from main import main
import netgen.meshing as ngmeshing
from ngsolve import CoefficientFunction, Integrate, Mesh

from Functions.Helper_Functions.count_prismatic_elements import count_prismatic_elements
from Functions.Saving.FtoS import FtoS
import shutil
import re

class bulk_sim_runner():
    
    def __init__(self, superclass):
        self.filelist = os.listdir('OCC_Geometry/')
        
        match superclass:
            case 'Jewellery':
                allowed_classes = ['Bracelet', 'Earring', 'Pendant', 'Ring', 'Watch']
            case 'Weapons':
                allowed_classes = ['KnuckleDuster', 'Knife']
            case 'Clothing':
                allowed_classes = ['BeltBuckle', 'ShoeShank']
        
        self.filelist = [sweep for sweep in self.filelist if sweep.split('_')[1] in allowed_classes]
        self.filelist = sorted(self.filelist)
        self.Nsims = len(self.filelist)
    
    def run_sims(self):
        
        for ind, sweep in enumerate(self.filelist):
            if os.path.isfile(f'VolFiles/{sweep[:-2]}' + 'vol'):
                print(f'Volfile for {sweep} already exists')
            else:
                print('\033[96m' + f'Running for {sweep}: {ind} / {self.Nsims}' + '\033[0m')
                output = main(geometry=sweep, start_stop=(0,10,81*3),order=4, use_POD=True, use_OCC=True)
            # output = 1
                #self.write_latex(sweep, output)

    def write_latex(self, superclass='Jewellery'):
        
        # match superclass:
        #     case 'Jewellery':
        #         allowed_classes = ['Bracelet', 'Earring', 'Pendant', 'Ring', 'Watch']
        #     case 'Weapons':
        #         allowed_classes = ['KnuckleDuster', 'Knife']
        
        # self._generate_preamble() 
        # self._write_global_settings()
        
        if os.path.exists('Object_Descriptions.tex'):
            os.remove('Object_Descriptions.tex')
        
        for sweep in self.filelist:
            
            # if sweep.split('_')[1] not in allowed_classes:
            #     continue
            
            # Computing sensible things. 
            try:
                mur, sig, inorout = self._get_materials(sweep)
                mesh = Mesh("VolFiles/" + sweep[:-2]+'vol')
            except: # can't load vol file
                continue
            mat_list, mur_dict, sig_dict, prism_elements, tet_elements, volume, skin_depth, tau = self.compute_object_info(sweep, mur, sig, inorout, mesh)                   
            
            # Writing
            self.write_object_description(sweep, mat_list, mur_dict, sig_dict, prism_elements, tet_elements, volume, skin_depth, tau)
                
        # # Closing file
        # with open(f"Doc.tex", 'a') as doc:
        #     lines = ['\n',
        #             '\n',
        #             r'\end{document}']
        #     doc.writelines(lines)

    def compute_object_info(self, sweep, mur, sig, inorout, mesh):
        mat_list = mesh.GetMaterials()
        mur_dict = dict(zip(mat_list, mur))
        sig_dict = dict(zip(mat_list, sig))
        inout = dict(zip(mat_list, inorout))
            
        prism_elements, tet_elements = count_prismatic_elements("VolFiles/" + sweep[:-2]+'vol')
            
            
        inout_coef = [inout[mat] for mat in mesh.GetMaterials()]
        inout = CoefficientFunction(inout_coef)
        volume = Integrate(inout, mesh) * (1e-3)**3
            
            # cleaning dictionaries
        mat_list = [i for i in mat_list if i!='air']
        {mat:mur_dict[mat] for mat in mat_list if mat!='air'}
        {mat:sig_dict[mat] for mat in mat_list if mat!='air'}
            
        skin_depth = [(2 / np.sqrt(1e10 * mur_dict[mat] * sig_dict[mat] * 4*np.pi*1e-7)) for mat in mat_list]
        tau = [x / 1e-3 for x in skin_depth]
        return mat_list,mur_dict,sig_dict,prism_elements,tet_elements,volume,skin_depth,tau

    def write_object_description(self, sweep, mat_list, mur_dict, sig_dict, prism_elements, tet_elements, volume, skin_depth, tau, superclass='Jewellery'):
        
        sweepname = f'Results/{sweep[:-3]}'
        sweepname += f'/al_0.001_mu_1_sig_{",".join(str(FtoS(sig_dict[x])) for x in mat_list)}'
        sweepname += f'/1-1e10_243_el_{tet_elements+prism_elements}_ord_4_POD_13_1e-6/Graphs/'  
        
        classname = sweep.split('_')[1]
        formattedpath = fr'Class_{superclass}/Class_{classname}/OBJ_{"_".join(sweep[:-3].split("_")[2:-1])}'
        formattedpath += fr'/al_0.001_mu_1_sig_{",".join(str(FtoS(sig_dict[x])) for x in mat_list)}'

        
        with open(f"texFiles/{sweep[:-3]}.tex", "w") as f:
            lines = [r'\begin{table}[!h]',
                        '\n',
                        r'\begin{tabular}{l | l}',
                        fr'Name             & {sweep[:-3]} \\'.replace('_', r'\_'), '\n',
                        fr'Geofile          & {"_".join(sweep[4:-3].split("_")[1:-1])}.geo \\'.replace('_', r'\_'), '\n'
                        fr'Class            & {superclass}, {classname}\\', '\n', 
                        fr'Formatted Path   & {formattedpath} \\'.replace('_', r'\_'), r'\hline', '\n'
                        fr'Volume [m$^3$]   & {volume} \\', '\n',
                        fr'Materials        & {",".join(str(x) for x in mat_list)}\\', '\n',
                        fr'$\mu_r$          & {",".join(str(mur_dict[x]) for x in mat_list)}\\', '\n',
                        fr'$\sigma_*$ [S/m] & {",".join(str(sig_dict[x]) for x in mat_list)}\\', '\n',
                        fr'Skin Depth [m]   & {",".join(str(x) for x in skin_depth)}\\', '\n',
                        fr'$\tau$           & {",".join(str(x) for x in tau)} \\', '\n',
                        fr'$N$ Tets         & {tet_elements} \\','\n', 
                        fr'$N$ Prisms       & {prism_elements}', '\n',
                        r'\end{tabular}', '\n',
                        r'\end{table}', '\n\n',
                        r'\begin{figure}[!h]', '\n',
                        r'$\begin{array}{c c}', '\n',
                        rf'\includegraphics[width=0.45\textwidth]{{{sweepname}RealTensorCoeficients}} & \includegraphics[width=0.45\textwidth]{{{sweepname}ImaginaryTensorCoeficients}} \\', 
                        '\n',
                        '(a) & (b)\n',
                        
                        r'\end{array}$', '\n',
                        rf'\caption{{{sweep[:-3]}: ($a$) real and ($b$) imaginary tensor coeficients.}}'.replace("_","\_"),
                        r'\end{figure}'                    
                        
                        ]
            


            f.writelines(lines)

        
        with open(f"Object_Descriptions.tex", 'a') as doc:
            
            lines = [rf'\subsection{{{sweep[:-3]}}}'.replace('_', '\_'),
                        '\n',
                        rf'\input{{texFiles/{sweep[:-3]}}}', '\n',
                        r'\clearpage', '\n']
            doc.writelines(lines)

    def _generate_preamble(self):
        with open(f"Doc.tex", 'w') as doc:
            lines = [r'\documentclass[a4paper,12]{article}', '\n',
                    r'\usepackage{amsmath, amsfonts, amssymb, amsthm, bm, graphics, bbm, color}', '\n',
                    r'\usepackage{graphicx}', '\n',
                    r'% redefine paper size', '\n',
                    r'\setlength{\oddsidemargin}{0in}', '\n',
                    r'\setlength{\textwidth}{6.4in}', '\n',
                    r'\setlength{\topmargin}{-0.5in}','\n', 
                    r'\setlength{\textheight}{9.9in}', '\n',
                    r'\setlength{\headheight}{0in}', '\n',
                    '\n\n',
                    '\n\n',
                    r'\begin{document}',
                    '\n\n',
                    r'\title{Documentation for MPT-Library 2.0}',
                    '\n\n',
                    r'\author{J. Elgy and P.D. Ledger\\', '\n', 
                    r'School of Computer Science and Mathematics, Keele University\\', '\n',
                    r'Keele, Staffordshire,  ST5 5BG.  U.K\\', '\n',
                    r'corresponding author: j.elgy@keele.ac.uk}', '\n',
                    r'\maketitle', '\n',
                    r'\tableofcontents',
                    '\n\n'
                    ]
            doc.writelines(lines)
                
    def _write_global_settings(self):
        with open(f"Doc.tex", 'a') as doc:
            lines = ['\n\n',
                    r'\section{Global Simulation Settings}', '\n\n',
                    r'\begin{table}[!h]', '\n',
                    r'\begin{tabular}{l | l}'
                    r'\multicolumn{2}{l}{Global Simulation Settings}  \\', '\n',
                    r'\hline', '\n',
                    r'Order                      & 4                  \\', '\n',
                    r'$\min{\omega}$ [rad/s]     & 1                  \\', '\n',
                    r'$\max{\omega}$ [rad/s]     & $10^{10}$          \\', '\n',
                    r'$N_{output}$               & 243                \\', '\n',
                    r'$N_{snapshots}$            & 13                 \\', '\n',
                    r'$TOL_\Sigma$               & $10^{-6}$          \\', '\n',
                    r'$TOL$                      & $10^{-8}$          \\', '\n',
                    r'$\epsilon$                 & $10^{-10}$         \\', '\n',
                    r'Postprocessing Method      & Matrix Method (MM) \\', '\n',
                    r'Preconditioner             & BDDC               \\', '\n',
                    r'Max Iterations             & 1500', '\n',
                    r'\end{tabular}', '\n',
                    r'\end{table}', '\n'
                    r'\clearpage', '\n\n',
                    r'\section{Objects}', '\n\n'
                    ]
            doc.writelines(lines)
    
    def _get_materials(self, sweep):
        Geometry = sweep[:-2] + 'geo'
        matlist = []
        orderedmatlist = []
        murlist = []
        siglist = []
        inout = []
        condlist=[]
 
        
            # Read the .geo file
        f = open("GeoFiles/" + Geometry, "r")
        f1 = f.readlines()
        for line in f1:
            # Search for lines where a top level object has been defined
            if line[:3] == "tlo":
                # find the materials and save them in the list
                # Find where the material name starts
                place = line.find("#")
                # Find where the end of the material name is
                if line[-1:] == "\n":
                    matend = line.find(" ", place)
                    mat = line[place + 1:matend]
                else:
                    if line.find(" ", place) != -1:
                        matend = line.find(" ", place)
                        mat = line[place + 1:matend]
                    else:
                        mat = line[place + 1:]
                # Add the material name to the list
                orderedmatlist.append(mat)
                # Check whether we've found this material before
                if orderedmatlist.count(mat) == 1 and mat != "air":
                    # find the properites for the materials
                    # Check how the line ends
                    if line[-1:] == "\n":
                        # Check if the line ends "_\n"
                        if line[-2] == " ":
                            if line.find("-mur=") != -1:
                                murplace = line.find("-mur=")
                                murend = line.find(" ", murplace)
                                mur = float(line[murplace + 5:murend])
                                murlist.append(mur)
                            if line.find("-sig=") != -1:
                                sigplace = line.find("-sig=")
                                sigend = line.find(" ", sigplace)
                                sig = float(line[sigplace + 5:sigend])
                                siglist.append(sig)
                        # Line ends in some sort of information
                        else:
                            if line.find("-mur=") != -1:
                                murplace = line.find("-mur=")
                                murend = line.find(" ", murplace)
                                mur = float(line[murplace + 5:murend])
                                murlist.append(mur)
                            if line.find("-sig=") != -1:
                                sigplace = line.find("-sig=")
                                sigend = line.find("\n", sigplace)
                                sig = float(line[sigplace + 5:sigend])
                                siglist.append(sig)
                    # must be the last line in the script but ends in a space
                    elif line[len(line) - 1] == " ":
                        if line.find("-mur=") != -1:
                            murplace = line.find("-mur=")
                            murend = line.find(" ", murplace)
                            mur = float(line[murplace + 5:murend])
                            murlist.append(mur)
                        if line.find("-sig=") != -1:
                            sigplace = line.find("-sig=")
                            sigend = line.find(" ", sigplace)
                            sig = float(line[sigplace + 5:sigend])
                            siglist.append(sig)
                    # must be the last line in the script but ends in some sort of information
                    else:
                        if line.find("-mur=") != -1:
                            murplace = line.find("-mur=")
                            murend = line.find(" ", murplace)
                            mur = float(line[murplace + 5:murend])
                            murlist.append(mur)
                        if line.find("-sig=") != -1:
                            sigplace = line.find("-sig=")
                            sig = float(line[sigplace + 5:])
                            siglist.append(sig)
                elif orderedmatlist.count(mat) == 1 and mat == "air":
                    murlist.append(1)
                    siglist.append(0)

        # Reorder the list so each material just appears once
        for mat in orderedmatlist:
            if mat not in matlist:
                matlist.append(mat)
        # decide in or out
        for mat in matlist:
            if mat == "air":
                inout.append(0)
            else:
                inout.append(1)

        return murlist, siglist, inout


def autoformat_results(superclass='Jewellery'):
    
    print('Autoformatting results directory into MPT-Classifier format.')
    
    # making folders:
    if os.path.isdir(f'Class_{superclass}') is False:
        os.mkdir(f'Class_{superclass}')
    
    classes = []
    for files in os.listdir('Results'):
        classes += [files.split('_')[1]]
    classes = set(classes)
    for classname in classes:
        if os.path.isdir(f'Class_{superclass}/Class_{classname}') == False:
            os.mkdir(f'Class_{superclass}/Class_{classname}')
        
        
    # Getting object names and copying to appropriate directory:
    for files in sorted(os.listdir('Results')):
        objectname = ['_'.join(files.split('_')[2:-1])]
        classname = [files.split('_')[1]]
        foldername = f'Class_{superclass}/Class_{classname[0]}/OBJ_{objectname[0]}'
        shutil.copytree(f'Results/{files}', foldername, dirs_exist_ok=True)

        # Removing sweeps that don't match the global parameters. E.g order=0 test runs.
        for children in os.listdir(foldername):
            for folder in os.listdir(foldername + '/' + children):
                folderparts = folder.split('_')

                if (folderparts[0] == '1-1e10')  and (folderparts[1] == '243') and (folderparts[5] == '4') and (folderparts[7] == '13') and (folderparts[8] == '1e-6'):
                    pass
                else:
                    print(f'Sweepname ({foldername}/{children}/{folder}) did not match expected parameters. Removing.')
                    shutil.rmtree(f'{foldername}/{children}/{folder}') 
    




     
if __name__ == '__main__':
    runner = bulk_sim_runner('Clothing')
    print(runner.filelist)
    runner.run_sims()
    #runner.write_latex()
    
    # autoformat_results()