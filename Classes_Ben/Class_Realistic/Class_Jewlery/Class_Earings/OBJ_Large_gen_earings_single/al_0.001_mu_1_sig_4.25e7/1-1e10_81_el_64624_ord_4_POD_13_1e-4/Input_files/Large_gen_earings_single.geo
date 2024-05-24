algebraic3d

solid boxout = orthobrick (-1000, -1000, -1000; 1000, 1000, 1000);

#Create the right earing
solid right_gem_cuff = (plane(0,0,1;0,0,1)
	and ellipsoid(7,100,0;7,0,0;0,6,0;0,0,4)
	and not ellipsoid(7,100,0;6,0,0;0,5,0;0,0,3))
	or torus(-0.75,100,0;0,0,1;1;0.35);

solid right_hoop = ellipticcylinder(-2.75,100,0;1.5,0,0;0,0,1)
	and plane(0,100.5,0;0,1,0)
	and plane(0,99.5,0;0,-1,0)
	and not ellipticcylinder(-2.75,100,0;1,0,0;0,0,0.5);

solid right_loop_protect = plane(-2.75,0,0;1,0,0)
	and plane(0,100.5,0;0,1,0)
	and plane(0,99.5,0;0,-1,0)
	and not ellipticcylinder(-2.75,100,0;1,0,0;0,0,0.5);

solid right_loop_front = right_loop_protect
	and plane(0,0,-4;0,0,-1)
	and ellipticcylinder(-2.75,0,-4;10,0,0;0,0,4.9)
	and not ellipticcylinder(-5.75,0,-4;6.5,0,0;0,0,3.9);

solid right_loop_back = plane(0,0,-4;0,0,1)
	and plane(-2.75,0,0;1,0,0)
	and plane(0,100.5,0;0,1,0)
	and plane(0,99.5,0;0,-1,0)
	and ellipticcylinder(-2.75,0,-4;10,0,0;0,0,4.9)
	and not ellipticcylinder(-2.75,0,-4;9.5,0,0;0,0,4.4);

	

solid right_earing = right_gem_cuff or right_hoop or right_loop_front or right_loop_back;
solid earings = right_earing-maxh=1;

solid rest = boxout and not earings;

tlo rest -transparent -col=[0,0,1];#air
tlo earings -col=[1,0.25,0.25];#ring -mur=1 -sig=4.03E+07