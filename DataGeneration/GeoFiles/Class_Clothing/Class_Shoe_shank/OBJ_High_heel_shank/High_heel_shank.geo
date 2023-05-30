algebraic3d

solid boxout = orthobrick (-1000, -1000, -1000; 1000, 1000, 1000);


#Create the shank
solid flat_shank = orthobrick(0,-6,-0.5;50,6,0.5);

solid curve_shank = cylinder(50,0,-49.5;50,1,-49.5;50)
	and not cylinder(50,0,-49.5;50,1,-49.5;49)
	and plane(0,-6,0;0,-1,0)
	and plane(0,6,0;0,1,0)
	and plane(50,0,0;-1,0,0)
	and plane(90,0,-20.5;4,0,-3);

solid curve_shank2 = cylinder(110,0,-4.5;110,1,-4.5;26)
	and not cylinder(110,0,-4.5;110,1,-4.5;25)
	and plane(90,0,-20.5;-4,0,3)
	and plane(110,0,-40;1,0,0)
	and plane(0,-6,0;0,-1,0)
	and plane(0,6,0;0,1,0);



solid full_shank = flat_shank or curve_shank or curve_shank2-maxh=2;

solid rest = boxout and not full_shank;

tlo rest -transparent -col=[0,0,1];#air
tlo full_shank -col=[1,0.25,0.25];#shank -mur=1 -sig=4.03E+07