from netgen.csg import *


from netgen.occ import *

"""
James Elgy - 2023:
Two joined tetrahedra for Netgen OCC geometry loaded from existing step geometry.
"""

material_name = ['carbon_steel', 'copper']
sigma = [1.6e6, 58e6]
mur = [100, 1]
alpha = 0.001

# Setting Boundary layer Options:
max_target_frequency = 1e10
boundary_layer_material = material_name[0]
number_of_layers = 3


geo = CSGeometry(r'GeoFiles/Knife_Santoku.geo')

# geo.GetSolids()[0].mat('air')
# geo.GetSolids()[1].mat('default')
# geo.GetSolids()[2].mat('default')


nmesh = geo.GenerateMesh(meshsize.coarse, optsteps3d=5, grading=0.6)
nmesh.SetMaterial(1, 'air')
nmesh.SetMaterial(2, material_name[0])
nmesh.SetMaterial(3, material_name[0])
nmesh.SetMaterial(4, material_name[0])
nmesh.SetMaterial(5, material_name[1])
# nmesh.SetMaterial(3, 'tetra')

nmesh.SetBCName(0, 'outer')
nmesh.SetBCName(1, 'outer')
nmesh.SetBCName(2, 'outer')
nmesh.SetBCName(3, 'outer')
nmesh.SetBCName(4, 'outer')
nmesh.SetBCName(5, 'outer')
# nmesh.SetBCName(6, 'outer')


#print(help(geo))


# Applying Boundary Layers:
mu0 = 4 * 3.14159 * 1e-7
tau = (2/(max_target_frequency * sigma[0] * mu0 * mur[0]))**0.5 / alpha
layer_thicknesses = [(2**n)*tau for n in range(number_of_layers)]

nmesh.BoundaryLayer(boundary=".*", thickness=layer_thicknesses, material=boundary_layer_material, domains=boundary_layer_material, outside=False)

#print(layer_thicknesses)

nmesh.Save(r'VolFiles/CSG_Knife_Knife_Santoku_carbonsteel_copper_rivets.vol')