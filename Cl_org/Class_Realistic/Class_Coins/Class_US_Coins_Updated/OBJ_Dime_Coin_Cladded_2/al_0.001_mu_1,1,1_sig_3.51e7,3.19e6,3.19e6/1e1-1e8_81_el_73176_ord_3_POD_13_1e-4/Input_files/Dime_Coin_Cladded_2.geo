algebraic3d

solid boxout = orthobrick (-1000, -1000, -1000; 1000, 1000, 1000);
solid cylin_inner = cylinder ( 0, 0, -0.475; 0, 0, 0.475; 8.955 )
	and plane (0, 0, -0.475; 0, 0, -1)
	and plane (0, 0, 0.475; 0, 0, 1) -maxh=0.5;

solid cylin_top = cylinder ( 0, 0, 0.475; 0, 0, 0.675; 8.955 )
	and plane (0, 0, 0.475; 0, 0, -1)
	and plane (0, 0, 0.675; 0, 0, 1) -maxh=0.5;

solid cylin_bottom = cylinder ( 0, 0, -0.475; 0, 0, -0.675; 8.955 )
	and plane (0, 0, -0.675; 0, 0, -1)
	and plane (0, 0, -0.475; 0, 0, 1) -maxh=0.5;


solid rest = boxout and not cylin_top and not cylin_bottom and not cylin_inner;


tlo rest -transparent -col=[0,0,1];#air
tlo cylin_inner -col=[1,0,0];#coin -mur=1 -sig=35.1e6
tlo cylin_top -col=[1,0,1];#coin_cladding_top -mur=1 -sig=3.19e6
tlo cylin_bottom -col=[0,0,0];#coin_cladding_bottom -mur=1 -sig=3.19e6