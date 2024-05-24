algebraic3d

solid boxout = orthobrick (-1000, -1000, -1000; 1000, 1000, 1000);

#Create the earing
solid stud = (cylinder(0,0,0;1,0,0;0.6)
	and plane(0,0,0;-1,0,0)
	and plane(38,0,0;1,0,0))
	or sphere(0,0,0;1.5)
	or sphere(38,0,0;1.5);



solid earings = stud-maxh=0.5;

solid rest = boxout and not earings;

tlo rest -transparent -col=[0,0,1];#air
tlo earings -col=[1,0.25,0.25];#ring -mur=1 -sig=4.03E+07