########################################################################
#This code was written by Ben Wilson with the supervison of Paul Ledger
#at the ZCCE, Swansea University
#Powered by NETGEN/NGSolve
########################################################################

#Object of interest
Object = 'Ring_with_ball.vol'
#(String) volume file you would like to simulate

#BoxSize
Box = 1
#(Int) 1-3 size of the box 1 => 2m^3, 2 => 5m^3, 3 => 10^3









########################################################################
#Import
#from netgen.meshing import *
from ngsolve import *
import netgen.meshing as ngmeshing
from netgen.csg import *


#Main script

#Load the box
m1 = ngmeshing.Mesh(dim=3)
m1.Load('Box.vol')
m1 = Mesh('Box.vol')
#Load the object
m2 = ngmeshing.Mesh(dim=3)
m2.Load(Object)
m2 = Mesh(Object)


#Create the new mesh
mesh = ngmeshing.Mesh()

#Cerate the face descriptors
fd_outside = mesh.Add (ngmeshing.FaceDescriptor(bc=1,domin=1,surfnr=1))
fd_inside = mesh.Add (ngmeshing.FaceDescriptor(bc=2,domin=2,domout=1,surfnr=2))

#Copy the surface elements for the box
pmap1 = { }
for e in m1.Elements(BND):
    for v in e.vertices:
        if (v not in pmap1):
            pmap1[v] = mesh.Add (ngmeshing.MeshPoint(Pnt(m1[v].point)))

for e in m1.Elements(BND):
    mesh.Add (ngmeshing.Element2D (fd_outside, [pmap1[v] for v in e.vertices]))

#Copy the surface elements for the object
pmap2 = { }
for e in m2.Elements(BND):
    for v in e.vertices:
        if (v not in pmap2):
            pmap2[v] = mesh.Add (ngmeshing.MeshPoint(Pnt(m2[v].point)))

for e in m2.Elements(BND):
    mesh.Add (ngmeshing.Element2D (fd_inside, [pmap2[v] for v in e.vertices]))


#Add the volume elements for the object mesh
#pmap3 = { }
#for e in m2.Elements(VOL):
#    for v in e.vertices:
#        if (v not in pmap3):
#            pmap3[v] = mesh.Add (ngmeshing.MeshPoint(Pnt(m2[v].point)))

#for e in m2.Elements(VOL):
#    mesh.Add(ngmeshing.Element3D (3,[pmap3[v] for v in e.vertices]))


#Finally remesh the volume elements and save
mesh.GenerateVolumeMesh()
mesh.Save("output.vol")


