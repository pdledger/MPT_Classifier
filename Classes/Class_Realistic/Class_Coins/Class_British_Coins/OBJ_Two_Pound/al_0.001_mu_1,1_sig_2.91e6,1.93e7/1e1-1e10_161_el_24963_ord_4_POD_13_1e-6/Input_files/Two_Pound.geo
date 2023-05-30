algebraic3d

solid boxout = orthobrick (-1000, -1000, -1000; 1000, 1000, 1000);

solid innercut = cylinder ( 0, 0, -1.25; 0, 0, 1.25; 10.5 );
solid inner = innercut
	and plane (0, 0, -1.25; 0, 0, -1)
	and plane (0, 0, 1.25; 0, 0, 1) -maxh=1.1;
	
solid outercyl = cylinder ( 0, 0, -1.25; 0, 0, 1.25; 14.2 )
	and plane (0, 0, -1.25; 0, 0, -1)
	and plane (0, 0, 1.25; 0, 0, 1) -maxh=1.1;



solid outer = outercyl and not innercut;

solid rest = boxout and not inner and not outer;

tlo rest -transparent -col=[0,0,1];#air
tlo inner -transparent -col =[0,1,0];#inner -mur=1 -sig=2.91E+06
tlo outer -col=[1,0,0];#outer -mur=1 -sig=1.93E+07
