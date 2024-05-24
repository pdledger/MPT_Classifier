from netgen.csg import *
from netgen.occ import *


material_name = ['stainless_steel']
sigma = [1.450E+06]
mur = [67.5]
alpha = 0.001

# Setting Boundary layer Options:
max_target_frequency = 1e10
boundary_layer_material = material_name[0]
number_of_layers = 3


geo = CSGeometry(r'GeoFiles/Knife_Cheap_Cutlet.geo')


nmesh = geo.GenerateMesh(meshsize.very_coarse)
nmesh.SetMaterial(1, 'air')
nmesh.SetMaterial(2, material_name[0])
nmesh.SetMaterial(3, material_name[0])
nmesh.SetMaterial(4, material_name[0])

# Setting boundary condition name for outer boundary
for i in range(6):
    nmesh.SetBCName(i, 'outer')
    

# Applying Boundary Layers:
mu0 = 4 * 3.14159 * 1e-7
tau = (2/(max_target_frequency * sigma[0] * mu0 * mur[0]))**0.5 / alpha
layer_thicknesses = [(2**n)*tau for n in range(number_of_layers)]

nmesh.BoundaryLayer(boundary=".*", thickness=layer_thicknesses, material=boundary_layer_material, domains=boundary_layer_material, outside=False)


    
nmesh.Save(r'VolFiles/CSG_Knife_Knife_Cheap_Cutlet_stainlesssteel.vol')
