algebraic3d

solid boxout = orthobrick (-1000, -1000, -1000; 1000, 1000, 1000);


#rings should vary from 14-22.6mm diameter (size 3-13.5)

solid ring = cylinder(0,0,-2.5;0,0,2.5;12)
	and plane(0,0,2.5;0,0,1)
	and plane(0,0,-2.5;0,0,-1)
	and not cylinder(0,0,-1;0,0,1;10)-maxh=1;

solid rest = boxout and not ring;

tlo rest -transparent -col=[0,0,1];#air
tlo ring -col=[1,0.25,0.25];#ring -mur=1 -sig=4.03E+07


