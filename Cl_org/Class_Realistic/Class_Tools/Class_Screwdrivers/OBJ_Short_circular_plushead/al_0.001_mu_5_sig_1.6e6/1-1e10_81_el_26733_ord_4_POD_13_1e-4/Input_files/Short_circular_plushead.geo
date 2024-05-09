algebraic3d

solid boxout = orthobrick (-1000, -1000, -1000; 1000, 1000, 1000);


solid shaft = cylinder(0,0,0;0,0,1;2.5)
	and plane(0,0,0;0,0,-1)
	and plane(0,0,62.5;0,0,1);

solid end = ellipsoid(0,0,46.5;3,0,0;0,3,0;0,0,16.5)
	or plane(0,0,46.5;0,0,1);

solid end_cut = (plane(0.3,0,62.5;-8,0,-1)
		and plane(0,0.3,62.5;0,-8,-1))
	or (plane(-0.3,0,62.5;8,0,-1)
		and plane(0,0.3,62.5;0,-8,-1))
	or (plane(0.3,0,62.5;-8,0,-1)
		and plane(0,-0.3,62.5;0,8,-1))
	or (plane(-0.3,0,62.5;8,0,-1)
		and plane(0,-0.3,62.5;0,8,-1));


solid driver = shaft and end and not end_cut-maxh=1.5;
solid rest = boxout and not driver;

tlo rest -transparent -col=[0,0,1];#air
tlo driver -col=[1,0.25,0.25];#ring -mur=1 -sig=4.03E+07