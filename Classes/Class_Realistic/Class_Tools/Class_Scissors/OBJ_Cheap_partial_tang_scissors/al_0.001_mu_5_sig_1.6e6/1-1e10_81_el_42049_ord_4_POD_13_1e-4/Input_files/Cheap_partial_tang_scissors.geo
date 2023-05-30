algebraic3d

solid boxout = orthobrick (-1000, -1000, -1000; 1000, 1000, 1000);


#Create the blades
solid blades = ellipticcylinder(0,-10,0;150,0,0;0,20,0)
	and ellipticcylinder(0,10,0;150,0,0;0,20,0)
	and plane(0,0,-2.5;0,0,-1)
	and plane(0,0,2.5;0,0,1)
	and plane(-40,0,0;-1,0,0)
	and not (plane(0,-10,-2.6;0,-1,8)
		and plane(10,0,0;-1,0,0)
		and plane(0,0,0;0,0,1))
	and not (plane(0,10,2.6;0,1,-8)
		and plane(10,0,0;-1,0,0)
		and plane(0,0,0;0,0,-1));

solid back_cuts = (plane(-15,10,0;1,-1,0)
	and plane(0,0,-0.1;0,0,-1))
	or (plane(-15,-10,0;1,1,0)
	and plane(0,0,0.1;0,0,1))
	or plane(-30,0,0;1,0,0);

solid tangs = orthobrick(-100,5,-2.5;0,9.9,-0.1)
	or orthobrick(-100,-9.9,0.1;0,-5,2.5);




solid base_scissors = blades and not back_cuts;

solid scissors = base_scissors or tangs-maxh=2.5;

solid rest = boxout and not scissors;

tlo rest -transparent -col=[0,0,1];#air
tlo scissors -col=[1,0.25,0.25];#scissors -mur=1 -sig=4.03E+07