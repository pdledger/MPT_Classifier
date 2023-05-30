algebraic3d

solid boxout = orthobrick (-1000, -1000, -1000; 1000, 1000, 1000);

#Create the right earing
solid right_hoop = torus(0,100,0;0,1,0;9;3.5)
	and torus(0,100,0;0,1,0;14;3)
	and plane(0,101.75,0;0,1,0)
	and plane(0,98.25,0;0,-1,0)
	and (plane(0,0,0;-1,0,0)
		or plane(0,0,0;0,0,-1));

solid right_pin = cylinder(-11.9,100,0;-11.9,100,-1;0.5)
	and plane(0,0,0;0,0,1)
	and plane(0,0,-8;0,0,-1)-maxh=0.25;

	
solid left_hoop = torus(0,-100,0;0,1,0;9;3.5)
	and torus(0,-100,0;0,1,0;14;3)
	and plane(0,-101.75,0;0,-1,0)
	and plane(0,-98.25,0;0,1,0)
	and (plane(0,0,0;-1,0,0)
		or plane(0,0,0;0,0,-1));

solid left_pin = cylinder(-11.9,-100,0;-11.9,-100,-1;0.5)
	and plane(0,0,0;0,0,1)
	and plane(0,0,-8;0,0,-1)-maxh=0.25;




solid earings = right_hoop or right_pin or left_hoop or left_pin-maxh=0.75;


solid rest = boxout and not earings;

tlo rest -transparent -col=[0,0,1];#air
tlo earings -col=[1,0.25,0.25];#ring -mur=1 -sig=4.03E+07