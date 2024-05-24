algebraic3d

solid boxout = orthobrick (-1000, -1000, -1000; 1000, 1000, 1000);
solid cylin = cylinder ( 0, 0, -0.775; 0, 0, 0.775; 15.305 )
	and plane (0, 0, -0.775; 0, 0, -1)
	and plane (0, 0, 0.775; 0, 0, 1) -maxh=0.5;

solid cylin_top = cylinder ( 0, 0, 0.775; 0, 0, 1.075; 15.305 )
	and plane (0, 0, 0.775; 0, 0, -1)
	and plane (0, 0, 1.075; 0, 0, 1) -maxh=1;

solid cylin_bottom = cylinder ( 0, 0, -0.775; 0, 0, -1.075; 15.305 )
	and plane (0, 0, -1.075; 0, 0, -1)
	and plane (0, 0, -0.775; 0, 0, 1) -maxh=0.5;

solid rest = boxout and not cylin and not cylin_top and not cylin_bottom;

tlo rest -transparent -col=[0,0,1];#air
tlo cylin-col=[1,0,0];#coin -mur=1 -sig=35.1e6
tlo cylin_top-col=[1,0,1];#coin_top -mur=1 -sig=3.19e6
tlo cylin_bottom-col=[0,0,0];#coin_bottom -mur=1 -sig=3.19e6
