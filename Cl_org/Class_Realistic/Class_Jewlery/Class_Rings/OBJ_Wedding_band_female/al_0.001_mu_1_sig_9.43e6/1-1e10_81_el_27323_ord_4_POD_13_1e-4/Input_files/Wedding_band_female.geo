algebraic3d

solid boxout = orthobrick (-1000, -1000, -1000; 1000, 1000, 1000);


#rings should vary from 14-22.6mm diameter (size 3-13.5)

#defin the shape of the ring
solid ring = torus(0,0,0;0,0,1;6.2;3)
	and torus(0,0,0;0,0,1;12;4)
	and torus(0,0,0;0,0,1;8.5;1)-maxh=1;

solid rest = boxout and not ring;

tlo rest -transparent -col=[0,0,1];#air
tlo ring -col=[1,0.25,0.25];#ring -mur=1 -sig=4.03E+07