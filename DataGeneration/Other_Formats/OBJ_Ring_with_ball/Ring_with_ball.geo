algebraic3d

#Create the earing
solid stud = torus(0,0,0;0,0,1;3;0.5)
	or sphere(0,3,0;1.2);


solid earings = stud-maxh=0.25;

tlo earings -col=[1,0.25,0.25];#ring -mur=1 -sig=4.03E+07