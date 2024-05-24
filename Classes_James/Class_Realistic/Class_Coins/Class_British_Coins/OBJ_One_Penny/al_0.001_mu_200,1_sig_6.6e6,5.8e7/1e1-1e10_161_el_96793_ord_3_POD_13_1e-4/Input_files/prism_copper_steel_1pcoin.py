from netgen.occ import *
from netgen.meshing import IdentificationType

from ngsolve import ngsglobals
ngsglobals.msg_level = 2
# For 1p coin
# we don not need to remove the layer thickness any more as
# including the boundary layer does not change the volume
r = 10.15
h = 1.52
h_outer = 1000
r_outer = 1000

bounding_cylinder = Cylinder(Pnt(-h_outer/2,0,0), X, r=r_outer, h=h_outer)
cyl = Cylinder(Pnt(-h/2,0,0), X, r=r, h=h)

cyl.mat('coin')
cyl.bc('default')
cyl.maxh = 0.5
bounding_cylinder.bc('outer')
bounding_cylinder.mat('air')

cyl.faces.name="default"
cyl.solids.name="coin"
outer_region = bounding_cylinder - cyl


mesh = OCCGeometry(Glue([cyl, outer_region])).GenerateMesh()
mesh.BoundaryLayer(boundary="default",thickness=[0.025], material="copper",
                   domains="coin", outside=False)
mesh.Save('prism_copper_steel_1pcoin.vol')
