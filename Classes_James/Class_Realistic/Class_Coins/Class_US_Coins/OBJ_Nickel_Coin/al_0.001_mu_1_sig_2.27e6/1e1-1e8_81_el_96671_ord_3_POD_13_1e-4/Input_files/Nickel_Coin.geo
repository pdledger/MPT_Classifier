algebraic3d

solid boxout = orthobrick (-1000, -1000, -1000; 1000, 1000, 1000);
solid cylin = cylinder ( 0, 0, -0.975; 0, 0, 0.975; 10.605 )
	and plane (0, 0, -0.975; 0, 0, -1)
	and plane (0, 0, 0.975; 0, 0, 1) -maxh=0.5;

solid rest = boxout and not cylin;

tlo rest -transparent -col=[0,0,1];#air
tlo cylin-col=[1,0,0];#coin -mur=1 -sig=2.27E+06
