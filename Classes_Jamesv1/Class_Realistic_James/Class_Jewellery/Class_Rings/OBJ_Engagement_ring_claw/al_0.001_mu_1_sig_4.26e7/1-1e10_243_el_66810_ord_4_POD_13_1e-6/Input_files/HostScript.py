import numpy as np
import pylatex
import os
from main import main
import netgen.meshing as ngmeshing
from ngsolve import CoefficientFunction, Integrate, Mesh

from pylatex import Document, Section, Subsection, Tabular
from pylatex import Math, TikZ, Axis, Plot, Figure, Matrix, Alignat, NoEscape
from pylatex.utils import italic


class bulk_sim_runner():
    def __init__(self):
        self.filelist = os.listdir('OCC_Geometry/')
    
    
    def run_sims(self):
        
        for sweep in self.filelist:
            if os.path.isfile(f'VolFiles/{sweep[:-2]}' + 'vol'):
                print(f'Volfile for {sweep} already exists')
            else:
                print('\033[96m' + f'Running for {sweep}' + '\033[0m')
                output = main(geometry=sweep, start_stop=(0,10,81*3),order=4, use_POD=True, use_OCC=True)
            # output = 1
                # self.write_latex(sweep, output)

    def write_latex(self, sweep, output):
        
        # Computing sensible things.

        mur, sig, inorout = self._get_materials(sweep)
        
        mesh = Mesh("VolFiles/" + sweep[:-2]+'vol')
        mat_list = mesh.GetMaterials()
        mur_dict = dict(zip(mat_list, mur))
        sig_dict = dict(zip(mat_list, sig))
        inout = dict(zip(mat_list, inorout))
        
        
        
        inout_coef = [inout[mat] for mat in mesh.GetMaterials()]
        inout = CoefficientFunction(inout_coef)
        volume = Integrate(inout, mesh) * (1e-3)**3
        
        # cleaning dictionaries
        mat_list = [i for i in mat_list if i!='air']
        {mat:mur_dict[mat] for mat in mat_list if mat!='air'}
        {mat:sig_dict[mat] for mat in mat_list if mat!='air'}
        
        skin_depth = [(2 / np.sqrt(1e10 * mur_dict[mat] * sig_dict[mat] * 4*np.pi*1e-7)) for mat in mat_list]
        tau = [x / 1e-3 for x in skin_depth]

        
        # Writing:
        geometry_options = {
            "head": "40pt",
            "margin": "0.5in",
            "bottom": "0.6in",
            "includeheadfoot": False
            }
        doc = Document(geometry_options=geometry_options, default_filepath=f'texFiles/{sweep[:-3]}')
    
        # creating a pdf with title "the simple stuff"
        with doc.create(Section(f'{sweep[:-3]}')):
            with doc.create(Tabular('c|c')) as table:
                table.add_hline()
                table.add_row(('materials', f'{",".join(str(x) for x in mat_list)}'))
                table.add_row((NoEscape(r'\sigma \textrm{[S/m]}'), f'{",".join(str(sig_dict[x]) for x in mat_list)}'))
                table.add_row((NoEscape('\mu_r'), f'{",".join(str(mur_dict[x]) for x in mat_list)}'))
                table.add_row((NoEscape('Volume [m^3]'), f'{volume}'))
                table.add_row(('Skin Depth [m]', f'{",".join(str(x) for x in skin_depth)}'))
                table.add_row((NoEscape(r'\tau'), f'{",".join(str(x) for x in tau)}'))
                table.add_row(('N Elements', output['NElements']))
                table.add_row(('Ndof', output['NDOF']))
                


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
        
if __name__ == '__main__':
    runner = bulk_sim_runner()
    runner.run_sims()
