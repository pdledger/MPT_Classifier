algebraic3d

solid boxout = orthobrick (-1000, -1000, -1000; 1000, 1000, 1000);


#Create the face
solid watch_face = (cylinder(0,0,0;1,0,0;21)
	and plane(3,0,0;1,0,0)
	and plane(-3,0,0;-1,0,0))
	or (ellipticcylinder(0,0,0;0,35,0;0,0,18)
	and plane(3,0,0;1,0,0)
	and cylinder(-22.5,0,0;-22.5,0,1;30)
	and not (cylinder(-22.5,0,0;-22.5,0,1;24)
		and not plane(-3,0,0;-1,0,0))
	and not (plane(0,24,0;0,-1,0)
		or plane(0,-24,0;0,1,0)
		or (plane(0,0,10.5;0,0,1)
			and plane(0,0,-10.5;0,0,-1)
			and not cylinder(0,0,0;1,0,0;21)))
	and not plane(-6,0,0;1,0,0))-maxh=2.5;

solid dials = (cylinder(0,0,0;0,0,1;2.75)
	and plane(0,0,24;0,0,1)
	and plane(0,0,0;0,0,-1))
	or (cylinder(0,0,0;0,1,2;2)
	and plane(0,11,22;0,1,2)
	and plane(0,0,0;0,0,-1))
	or (cylinder(0,0,0;0,-1,2;2)
	and plane(0,-11,22;0,-1,2)
	and plane(0,0,0;0,0,-1))-maxh=0.5;




solid watch = watch_face or dials;

solid rest = boxout and not watch;

tlo rest -transparent -col=[0,0,1];#air
tlo watch -col=[1,0.25,0.25];#watch -mur=1 -sig=4.03E+07