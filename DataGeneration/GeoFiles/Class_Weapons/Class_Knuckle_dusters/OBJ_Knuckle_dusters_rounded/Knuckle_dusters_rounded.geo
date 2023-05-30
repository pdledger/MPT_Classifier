algebraic3d

solid boxout = orthobrick (-1000, -1000, -1000; 1000, 1000, 1000);

solid brass = (ellipticcylinder(-13,44,0;-7.5,16,0;14,7,0)
	or ellipticcylinder(13,44,0;7.5,16,0;-14,7,0)
	or ellipticcylinder(-38.5,36,0;-11,15,0;13,10,0)
	or ellipticcylinder(38.5,36,0;11,15,0;-13,10,0))
	and plane(0,0,3.5;0,0,1)
	and plane(0,0,-3.5;0,0,-1)
	and not ellipticcylinder(-13,44,0;-3.5,12,0;10,3,0)
	and not ellipticcylinder(13,44,0;3.5,12,0;-10,3,0)
	and not ellipticcylinder(-38.5,36,0;-7,11,0;9,6,0)
	and not ellipticcylinder(38.5,36,0;7,11,0;-9,6,0);

solid handle = cylinder(0,30,0;0,30,1;26)
	and plane(0,0,3.5;0,0,1)
	and plane(0,0,-3.5;0,0,-1)
	and not cylinder(0,30,0;0,30,1;20)
	and not plane(0,32,0;0,-1,0);

solid handle2 = ellipticcylinder(0,4,0;35,0,0;0,4.5,0)
	and plane(0,0,3.5;0,0,1)
	and plane(0,0,-3.5;0,0,-1);




solid brass_knuckles = brass or handle or handle2-maxh=2;
solid rest = boxout and not brass_knuckles;

tlo rest -transparent -col=[0,0,1];#air
tlo brass_knuckles -col=[1,0.25,0.25];#ring -mur=1 -sig=4.03E+07