algebraic3d

solid boxout = orthobrick (-1000, -1000, -1000; 1000, 1000, 1000);


#Create the shank
solid shank = plane(-60,0,0;-1,0,0)
	and plane(60,0,0;1,0,0)
	and plane(0,-5,0;0,-1,0)
	and plane(0,5,0;0,1,0)
	and plane(0,0,0.4;0,0,1)
	and not (cylinder(0,0,2;1,0,2;2)
			and plane(45,0,0;1,0,0)
			and plane(-30,0,0;-1,0,0))
	and not (cylinder(0,3.75,0.75;1,3.75,0.75;0.75)
				and plane(-20,0,0;1,0,0)
				and plane(-52.5,0,0;-1,0,0))
	and not (cylinder(0,-3.75,0.75;1,-3.75,0.75;0.75)
				and plane(-20,0,0;1,0,0)
				and plane(-52.5,0,0;-1,0,0))
	and not (plane(0,0,-0.4;0,0,1)
		and not (cylinder(0,0,1.2;1,0,1.2;2)
			and plane(45,0,0;1,0,0)
			and plane(-30,0,0;-1,0,0))
		and not (cylinder(0,3.75,-0.05;1,3.75,-0.05;0.75)
			and plane(-20,0,0;1,0,0)
			and plane(-52.5,0,0;-1,0,0))
		and not (cylinder(0,-3.75,-0.05;1,-3.75,-0.05;0.75)
			and plane(-20,0,0;1,0,0)
			and plane(-52.5,0,0;-1,0,0))
			);

#orthobrick(-60,-6,-0.4;60,6,0.4);

solid cuts = cylinder(50,0,0;50,0,1;2)
	or cylinder(-35,0,0;-35,0,1;2)
	or cylinder(-45,0,0;-45,0,1;2)
	or orthobrick(-75,-2,-10;-45,2,10);




solid full_shank = shank and not cuts-maxh=2;

solid rest = boxout and not full_shank;

tlo rest -transparent -col=[0,0,1];#air
tlo full_shank -col=[1,0.25,0.25];#shank -mur=1 -sig=4.03E+07