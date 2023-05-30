algebraic3d

solid boxout = orthobrick (-1000, -1000, -1000; 1000, 1000, 1000);


#Create the band
solid band = cylinder(0,0,-8;0,0,8;25)
	and ((plane(0,0,7;0,0,1)
		and plane(0,0,-7;0,0,-1))
	or (ellipticcylinder(0,0,0;0,22.5,0;0,0,13)
		and plane(0,0,0;-1,0,0)))
	and not (cylinder(-1.25,0,-5;-1.25,0,5;22.5)
		and not plane(18.5,0,0;-1,0,0))
	and not plane(21.5,0,0;-1,0,0);


solid watch = band-maxh=2.5;

solid rest = boxout and not watch;

tlo rest -transparent -col=[0,0,1];#air
tlo watch -col=[1,0.25,0.25];#watch -mur=1 -sig=4.03E+07