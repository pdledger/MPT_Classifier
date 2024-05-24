from netgen.csg import *
from netgen.occ import *


material_name = ['stainless_steel', 'naval_brass']
sigma = [1.450E+06, 1.508E+07]
mur = [67.5, 1]
alpha = 0.001

# Setting Boundary layer Options:
max_target_frequency = 1e10
boundary_layer_material = material_name[1]
number_of_layers = 3


geo = CSGeometry(r'GeoFiles/Flat_buckle.geo')


nmesh = geo.GenerateMesh(meshsize.very_coarse)
nmesh.SetMaterial(1, 'air')
nmesh.SetMaterial(2, material_name[0])

# Setting boundary condition name for outer boundary
for i in range(6):
    nmesh.SetBCName(i, 'outer')
    

# Applying Boundary Layers:
mu0 = 4 * 3.14159 * 1e-7
tau = (2/(max_target_frequency * sigma[0] * mu0 * mur[0]))**0.5 / alpha
# layer_thicknesses = [(2**n)*tau for n in range(number_of_layers)]

layer_thicknesses = [0.2 / 2] * 2 # 0.2mm thick brass plating. Taken from recommendations in https://prototypingsolutions.com/electroplating/

nmesh.BoundaryLayer(boundary=".*", thickness=layer_thicknesses, material=boundary_layer_material, domains=boundary_layer_material, outside=False)

nmesh.Save(r'VolFiles/CSG_BeltBuckle_Flat_buckle_brass_plated_stainlesssteel.vol')
