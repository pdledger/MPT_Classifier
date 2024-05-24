algebraic3d

solid boxout = orthobrick (-1000, -1000, -1000; 1000, 1000, 1000);


#Create the band
solid band = cylinder(-1.25,0,-8;-1.25,0,8;25)
	and ((plane(0,0,6.5;0,0,1)
		and plane(0,0,-6.5;0,0,-1))
	or (orthobrick(0,-12.5,-9;100,12.5,9)
		and plane(0,0,0;-1,0,0)))
	and not (cylinder(-1.25,0,-5;-1.25,0,5;23.5)
		and not plane(17.5,0,0;-1,0,0))
	and not plane(21.5,0,0;-1,0,0)-maxh=2.5;

solid dials = cylinder(19.5,0,0;19.5,0,1;1.75)
	and plane(0,0,10.5;0,0,1)
	and plane(0,0,0;0,0,-1)-maxh=0.5;


solid watch = band or dials;

solid rest = boxout and not watch;

tlo rest -transparent -col=[0,0,1];#air
tlo watch -col=[1,0.25,0.25];#watch -mur=1 -sig=4.03E+07