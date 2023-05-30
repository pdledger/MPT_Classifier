algebraic3d

solid boxout = orthobrick (-1000, -1000, -1000; 1000, 1000, 1000);


#rings should vary from 14-22.6mm diameter (size 3-13.5)

solid base_ring = cylinder(0,0,-8;0,0,8;12)
	and plane(0,0,8;0,0,1)
	and plane(0,0,-8;0,0,-1)
	and not cylinder(0,0,-5;0,0,5;10)
	and ellipticcylinder(0,0,0;0,12.5,0;0,0,5)
	and plane(11,0,0;1,0,0);

solid cut_1 = ellipticcylinder(-12,0,5;22.1,0,0;0,0,-3.5);
solid cut_2 = ellipticcylinder(-12,0,-5;22.1,0,0;0,0,3.5);

solid ring = base_ring and not cut_1 and not cut_2;

solid rest = boxout and not ring;

tlo rest -transparent -col=[0,0,1];#air
tlo ring -col=[1,0.25,0.25];#ring -mur=1 -sig=4.03E+07