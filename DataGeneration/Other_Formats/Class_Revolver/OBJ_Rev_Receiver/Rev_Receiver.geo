algebraic3d


solid boxout = orthobrick (-1000, -1000, -1000; 1000, 1000, 1000);


#Create the pretend gun
solid stud = torus(0,0,0;0,0,1;3;0.5)
	or sphere(0,3,0;1.2);


solid gun = stud-maxh=0.25;

solid rest = boxout and not gun;

tlo rest -transparent -col=[0,0,1];#air
tlo gun -col=[1,0.25,0.25];#gun -mur=5 -sig=4E+05