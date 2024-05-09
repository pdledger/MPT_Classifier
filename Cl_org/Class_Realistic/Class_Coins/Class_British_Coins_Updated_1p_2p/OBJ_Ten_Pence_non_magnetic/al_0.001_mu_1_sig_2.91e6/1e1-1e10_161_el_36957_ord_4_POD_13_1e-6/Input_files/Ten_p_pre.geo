algebraic3d

solid boxout = orthobrick (-1000, -1000, -1000; 1000, 1000, 1000);
solid cylin = cylinder ( 0, 0, -0.925; 0, 0, 0.925; 12.25 )
	and plane (0, 0, -0.925; 0, 0, -1)
	and plane (0, 0, 0.925; 0, 0, 1) -maxh=0.8;

solid rest = boxout and not cylin;

tlo rest -transparent -col=[0,0,1];#air
tlo cylin-col=[1,0.25,0.25];#coin -mur=1 -sig=2.91E+06
