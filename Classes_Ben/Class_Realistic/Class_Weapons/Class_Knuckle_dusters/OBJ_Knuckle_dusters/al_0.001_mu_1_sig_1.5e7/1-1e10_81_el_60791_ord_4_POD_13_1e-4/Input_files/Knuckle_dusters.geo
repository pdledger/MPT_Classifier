algebraic3d

solid boxout = orthobrick (-1000, -1000, -1000; 1000, 1000, 1000);

solid brass = orthobrick(-57.5,0,-3.5;57.5,60,3.5)
	and not ellipticcylinder(-13,44,0;-3.5,12,0;10,3,0)
	and not ellipticcylinder(13,44,0;3.5,12,0;-10,3,0)
	and not ellipticcylinder(-38.5,36,0;-7,11,0;9,6,0)
	and not ellipticcylinder(38.5,36,0;7,11,0;-9,6,0)
	and not (cylinder(0,30,0;0,30,1;22) and plane(0,28,0;0,1,0));

solid base = ellipticcylinder(0,34,0;55,0,0;0,33,0);

solid cuts = (plane(0,55,0;1,-2,0)
		and plane(0,55,0;-1,-2,0))
	or (plane(30,51,0;-1,-1,0)
		and plane(30,51,0;0,-1,0))
	or (plane(-30,51,0;1,-1,0)
		and plane(-35,51,0;0,-1,0))
	or orthobrick(27.5,12.5,-10;100,20,10)
	or orthobrick(-100,12.5,-10;-27.5,20,10);

solid brass_knuckles = brass and base and not cuts-maxh=2;


solid rest = boxout and not brass_knuckles;

tlo rest -transparent -col=[0,0,1];#air
tlo brass_knuckles -col=[1,0.25,0.25];#ring -mur=1 -sig=4.03E+07