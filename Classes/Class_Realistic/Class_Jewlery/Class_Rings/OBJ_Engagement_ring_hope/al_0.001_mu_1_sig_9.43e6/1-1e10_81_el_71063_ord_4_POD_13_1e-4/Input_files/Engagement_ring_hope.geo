algebraic3d

solid boxout = orthobrick (-1000, -1000, -1000; 1000, 1000, 1000);


#rings should vary from 14-22.6mm diameter (size 3-13.5)

solid base_ring = torus(0,0,0;0,0,1;8.5;0.625);

solid ring_gap = cylinder(0,0,0;10,0,0;1)
	and plane(0,0,0;-1,0,0);

solid diamond_gap = cylinder(0,0,0;10,0,0;1.5)
	and plane(7.5,0,0;-1,0,0)
	and plane(9.5,0,0;1,0,0)
	and not cylinder(0,0,0;10,0,0;0.75)-maxh=0.5;

solid ring = (base_ring or diamond_gap) and not ring_gap;

solid rest = boxout and not ring;

tlo rest -transparent -col=[0,0,1];#air
tlo ring -col=[1,0.25,0.25];#ring -mur=1 -sig=4.03E+07