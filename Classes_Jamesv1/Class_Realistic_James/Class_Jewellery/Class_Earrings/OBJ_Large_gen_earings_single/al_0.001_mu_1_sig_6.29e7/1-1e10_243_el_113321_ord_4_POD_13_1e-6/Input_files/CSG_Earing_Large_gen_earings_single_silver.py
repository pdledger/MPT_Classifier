from netgen.csg import *
from netgen.occ import *
import pylatex

material_name = ['silver']
sigma = [6.287E+07]
mur = [1]
alpha = 0.001

# Setting Boundary layer Options:
max_target_frequency = 1e8
boundary_layer_material = material_name[0]
number_of_layers = 3


geo = CSGeometry(r'GeoFiles/Large_gen_earings_single.geo')


nmesh = geo.GenerateMesh(meshsize.coarse)
nmesh.SetMaterial(1, 'air')
nmesh.SetMaterial(2, material_name[0])

# Setting boundary condition name for outer boundary
for i in range(6):
    nmesh.SetBCName(i, 'outer')
    
nmesh.Save(r'VolFiles/CSG_Earing_Large_gen_earings_single_silver.vol')
