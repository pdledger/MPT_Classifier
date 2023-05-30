algebraic3d

solid boxout = orthobrick (-1000, -1000, -1000; 1000, 1000, 1000);

#Create the right earing
solid right_stud = (cylinder(0,100,0;0,100,-1;2.6)
		and plane(0,0,0;0,0,-1)
		and plane(0,0,2.5;0,0,1))
	or (cylinder(0,100,0;0,100,-1;0.4)
		and plane(0,0,0;0,0,1)
		and plane(0,0,-10;0,0,-1))
	or (cylinder(0,100,0;0,100,-1;3)
		and plane(0,0,-6;0,0,1)
		and plane(0,0,-6.5;0,0,-1));

solid right_cutout = (cylinder(0,100,0;0,100,-1;2.1)
		and plane(0,0,0.5;0,0,-1)
		and plane(0,0,3.5;0,0,1))
	or orthobrick(0.25,100.25,0.5;10,110,1)
	or orthobrick(0.25,90,0.5;10,99.75,1)
	or orthobrick(-10,100.25,0.5;-0.25,110,1)
	or orthobrick(-10,90,0.5;-0.25,99.75,1);


solid left_stud = (cylinder(0,-100,0;0,-100,-1;2.6)
		and plane(0,0,0;0,0,-1)
		and plane(0,0,2.5;0,0,1))
	or (cylinder(0,-100,0;0,-100,-1;0.4)
		and plane(0,0,0;0,0,1)
		and plane(0,0,-10;0,0,-1))
	or (cylinder(0,-100,0;0,-100,-1;3)
		and plane(0,0,-6;0,0,1)
		and plane(0,0,-6.5;0,0,-1));

solid left_cutout = (cylinder(0,-100,0;0,-100,-1;2.1)
		and plane(0,0,0.5;0,0,-1)
		and plane(0,0,3.5;0,0,1))
	or orthobrick(0.25,-110,0.5;10,-100.25,1)
	or orthobrick(0.25,-99.75,0.5;10,-90,1)
	or orthobrick(-10,-110,0.5;-0.25,-100.25,1)
	or orthobrick(-10,-99.75,0.5;-0.25,-90,1);


solid right = right_stud and not right_cutout-maxh=0.3;
solid left = left_stud and not left_cutout-maxh=0.3;
solid earings = right or left;

solid rest = boxout and not earings;

tlo rest -transparent -col=[0,0,1];#air
tlo earings -col=[1,0.25,0.25];#ring -mur=1 -sig=4.03E+07