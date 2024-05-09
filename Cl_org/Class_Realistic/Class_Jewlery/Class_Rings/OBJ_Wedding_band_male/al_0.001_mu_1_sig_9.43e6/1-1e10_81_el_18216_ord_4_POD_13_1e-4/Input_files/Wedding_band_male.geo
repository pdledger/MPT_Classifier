algebraic3d

solid boxout = orthobrick (-1000, -1000, -1000; 1000, 1000, 1000);


#rings should vary from 14-22.6mm diameter (size 3-13.5)

#defin the shape of the ring
solid ring = torus(0,0,0;0,0,1;9.25;2.75)
	and torus(0,0,0;0,0,1;15;5)
	and torus(0,0,0;0,0,1;10.75;2.25)-maxh=1;

solid rest = boxout and not ring;

tlo rest -transparent -col=[0,0,1];#air
tlo ring -col=[1,0.25,0.25];#ring -mur=1 -sig=4.03E+07