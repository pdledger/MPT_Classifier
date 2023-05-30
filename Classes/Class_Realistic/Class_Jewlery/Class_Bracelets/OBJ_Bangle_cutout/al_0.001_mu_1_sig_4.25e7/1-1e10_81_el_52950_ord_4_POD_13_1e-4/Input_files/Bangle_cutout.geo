algebraic3d

solid boxout = orthobrick (-1000, -1000, -1000; 1000, 1000, 1000);



solid base_bangle = cylinder(0,0,-3.5;0,0,3.5;30)
	and plane(0,0,3.5;0,0,1)
	and plane(0,0,-3.5;0,0,-1)
	and ellipsoid(0,0,0;30.1,0,0;0,30.1,0;0,0,12.5)
	and not cylinder(0,0,-3.5;0,0,3.5;28)-maxh=1;

solid cut_out = plane(0,0,0;-1,-1.25,0)
	and plane(0,0,0;-1,1.25,0)
	and not cylinder(0,0,0;1.25,1,0;6)
	and not cylinder(0,0,0;1.25,-1,0;6);
	
solid bangle = base_bangle and not cut_out;

solid rest = boxout and not bangle;

tlo rest -transparent -col=[0,0,1];#air
tlo bangle -col=[1,0.25,0.25];#ring -mur=1 -sig=4.03E+07