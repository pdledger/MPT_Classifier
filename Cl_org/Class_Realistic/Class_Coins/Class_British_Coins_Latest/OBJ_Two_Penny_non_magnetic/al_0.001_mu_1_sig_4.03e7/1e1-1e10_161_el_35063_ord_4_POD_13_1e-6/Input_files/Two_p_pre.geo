algebraic3d

solid boxout = orthobrick (-1000, -1000, -1000; 1000, 1000, 1000);
solid cylin = cylinder ( 0, 0, -1.015; 0, 0, 1.015; 12.95 )
	and plane (0, 0, -1.015; 0, 0, -1)
	and plane (0, 0, 1.015; 0, 0, 1) -maxh=1;

solid rest = boxout and not cylin;

tlo rest -transparent -col=[0,0,1];#air
tlo cylin-col=[1,0.25,0.25];#coin -mur=1 -sig=4.03E+07
