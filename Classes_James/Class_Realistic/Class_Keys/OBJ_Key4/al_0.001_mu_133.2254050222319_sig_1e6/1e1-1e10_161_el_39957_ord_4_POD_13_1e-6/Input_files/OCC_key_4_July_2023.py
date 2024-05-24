from netgen.occ import *
from ngsolve import ngsglobals

#ngsglobals.msg_level = 3

material_name = ['steel']
mur=[133.2254050222319]
sigma = [1e6]
alpha = 1e-3

filename = 'StepFiles/Key4_030723_smooth_2.step'
geo = OCCGeometry(filename)
geo = geo.shape.Move((-geo.shape.solids[0].center.x, -geo.shape.solids[0].center.y, -geo.shape.solids[0].center.z))


geo.bc('default')
geo.mat(material_name[0])
geo.maxh = 1000

bounding_box = Box(Pnt(-1000, -1000, -1000), Pnt(1000, 1000, 1000))
bounding_box.mat('air')
bounding_box.bc('outer')

geo2 = OCCGeometry(Glue([geo, bounding_box]))

nmesh = geo2.GenerateMesh(minh=5, optsteps3d=10, minedgelen=0.2)


max_target_frequency = 1e10
number_of_layers = 2
mu0 = 4 * 3.14159 * 1e-7
tau = (2/(max_target_frequency * sigma[0] * mu0 * mur[0]))**0.5 / alpha
layer_thicknesses = [(2**n)*tau for n in range(number_of_layers)]

nmesh.BoundaryLayer(boundary='default', thickness=layer_thicknesses, material=material_name[0],
                           domains=material_name[0], outside=False)

print(mur)

nmesh.Save('VolFiles/OCC_key_4_July_2023.vol')
