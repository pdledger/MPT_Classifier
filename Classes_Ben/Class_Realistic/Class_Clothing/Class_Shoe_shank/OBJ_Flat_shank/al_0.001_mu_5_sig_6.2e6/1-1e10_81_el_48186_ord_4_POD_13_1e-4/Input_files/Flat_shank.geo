algebraic3d

solid boxout = orthobrick (-1000, -1000, -1000; 1000, 1000, 1000);


#Create the shank
solid shank = orthobrick(-60,-6,-0.4;60,6,0.4);

solid cuts = cylinder(50,0,0;50,0,1;2)
	or cylinder(-35,0,0;-35,0,1;2)
	or cylinder(-45,0,0;-45,0,1;2)
	or orthobrick(-75,-2,-10;-45,2,10);




solid full_shank = shank and not cuts-maxh=2;

solid rest = boxout and not full_shank;

tlo rest -transparent -col=[0,0,1];#air
tlo full_shank -col=[1,0.25,0.25];#shank -mur=1 -sig=4.03E+07