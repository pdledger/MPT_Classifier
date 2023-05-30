algebraic3d

solid boxout = orthobrick (-1000, -1000, -1000; 1000, 1000, 1000);



solid bangle = torus(0,0,0;0,0,1;32;2)
	and torus(0,0,0;0,0,1;30;1.5)
	and torus(0,0,0;0,0,1;31;1.25)-maxh=1;

solid rest = boxout and not bangle;

tlo rest -transparent -col=[0,0,1];#air
tlo bangle -col=[1,0.25,0.25];#ring -mur=1 -sig=4.03E+07