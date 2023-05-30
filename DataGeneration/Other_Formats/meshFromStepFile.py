#**************************************************************************
#  
# DESCRIPTION
# Generate .vol mesh from .step file
#
# HISTORY
# A. Amad     06/2020: code implementation
#
#**************************************************************************
import numpy as np
from ngsolve import *
from netgen.occ import *

nameMesh = "coneJohn.step"

#import step file geometry
geo = OCCGeometry(nameMesh)
geo.Glue()

mesh = Mesh(geo.GenerateMesh(meshsize.coarse))

print(mesh.GetBoundaries())
print(mesh.GetMaterials())


#save the mesh
mesh.ngmesh.Save ("newMesh.vol")


 
