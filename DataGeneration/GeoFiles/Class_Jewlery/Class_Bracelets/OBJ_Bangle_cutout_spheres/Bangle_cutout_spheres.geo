algebraic3d

solid boxout = orthobrick (-1000, -1000, -1000; 1000, 1000, 1000);



solid base_bangle = cylinder(0,0,-3.5;0,0,3.5;30)
	and plane(0,0,2.5;0,0,1)
	and plane(0,0,-2.5;0,0,-1)
	and ellipsoid(0,0,0;30.05,0,0;0,30.05,0;0,0,10)
	and not cylinder(0,0,-3.5;0,0,3.5;28);

solid cut_out = plane(0,0,0;-1,-1.25,0)
	and plane(0,0,0;-1,1.25,0);

solid spheres = sphere(23,18,0;4)
	or sphere(23,-18,0;4);
	
solid bangle = (base_bangle and not cut_out) or spheres-maxh=1;

solid rest = boxout and not bangle;

tlo rest -transparent -col=[0,0,1];#air
tlo bangle -col=[1,0.25,0.25];#ring -mur=1 -sig=4.03E+07