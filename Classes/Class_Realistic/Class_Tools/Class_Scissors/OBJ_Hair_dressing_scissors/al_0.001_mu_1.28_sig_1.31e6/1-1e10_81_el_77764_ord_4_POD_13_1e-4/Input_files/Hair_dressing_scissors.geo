algebraic3d

solid boxout = orthobrick (-1000, -1000, -1000; 1000, 1000, 1000);


#Create the blades
solid blades = ellipticcylinder(0,-35,0;150,0,0;0,40,0)
	and ellipticcylinder(0,35,0;150,0,0;0,40,0)
	and plane(0,0,-1.25;0,0,-1)
	and plane(0,0,1.25;0,0,1)
	and plane(-40,0,0;-1,0,0)
	and not (plane(0,-5,-1.3;0,-1,8)
		and plane(10,0,0;-1,0,0)
		and plane(0,0,0;0,0,1))
	and not (plane(0,5,1.3;0,1,-8)
		and plane(10,0,0;-1,0,0)
		and plane(0,0,0;0,0,-1));

solid back_cuts = (plane(0,10,0;1,-1,0)
	and plane(0,0,-0.1;0,0,-1))
	or (plane(0,-10,0;1,1,0)
	and plane(0,0,0.1;0,0,1))
	or plane(-30,0,0;1,0,0);



solid tang1 = cone(10,1.25,-0.25;3;-50,9,-0.25;1)
#cylinder(10,1.25,-0.25;-50,9,-0.25;1.25)
	and plane(5,0,0;1,0,0)
	and plane(-40,0,0;-1,0,0)
	and (plane(-20,0,-1.75;0.75,0,-20)
		or plane(0,0,-1.25;0,0,-1))
	and (plane(-15,0,1.25;1,0,4)
		or plane(0,0,-0.1;0,0,1));

solid tang2 = cone(10,-1.25,0.25;3;-50,-9,0.25;1)
#cylinder(10,-1.25,0.25;-50,-9,0.25;1.25)
	and plane(5,0,0;1,0,0)
	and plane(-40,0,0;-1,0,0)
	and (plane(-20,0,1.75;0.75,0,20)
		or plane(0,0,1.25;0,0,1))
	and (plane(-15,0,-1.25;1,0,-4)
		or plane(0,0,0.1;0,0,-1));

solid hoops = torus(-50.5,12.5,-0.25;0,0,1;11.6;1.35)
	or torus(-50.5,-12.5,-0.25;0,0,1;11.6;1.35);





solid tangs = tang1 or tang2;
solid base_scissors = blades and not back_cuts;

solid scissors = base_scissors or tangs or hoops-maxh=2.5 -minh=0.25;

solid rest = boxout and not scissors;

tlo rest -transparent -col=[0,0,1];#air
tlo scissors -col=[1,0.25,0.25];#scissors -mur=1 -sig=4.03E+07