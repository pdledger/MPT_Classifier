algebraic3d

solid boxout = orthobrick (-1000, -1000, -1000; 1000, 1000, 1000);
solid cylin = cylinder ( 0, 0, -0.775; 0, 0, 0.775; 9.525 )
	and plane (0, 0, -0.775; 0, 0, -1)
	and plane (0, 0, 0.775; 0, 0, 1) -maxh=0.5;

solid rest = boxout and not cylin;

tlo rest -transparent -col=[0,0,1];#air
tlo cylin-col=[1,0,0];#coin -mur=1 -sig=15.9E+06
