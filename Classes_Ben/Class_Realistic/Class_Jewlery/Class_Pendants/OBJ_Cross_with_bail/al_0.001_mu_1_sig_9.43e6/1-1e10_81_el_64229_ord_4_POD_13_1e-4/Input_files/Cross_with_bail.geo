algebraic3d

solid boxout = orthobrick (-1000, -1000, -1000; 1000, 1000, 1000);


solid base_pendant = orthobrick(0,-1,-0.75;19,1,0.75)
	or orthobrick(5,-6,-0.75;7,6,0.75)
	or torus(-0.75,0,0;0,0,1;1;0.35);

solid bail = ((plane(-6,0,0;-1,0,0)
	and ellipticcylinder(-6,0,0;5.5,0,0;0,0,1.25)
	and not ellipticcylinder(-6,0,0;4.5,0,0;0,0,0.75))
	or (plane(-6,0,0;1,0,0)
		and cylinder(-6,0,0;-6,1,0;1.25)
		and not cylinder(-6,0,0;-6,1,0;0.75)))
	and ellipticcylinder(-9.5,0,0;8.9,0,0;0,1.25,0);

solid pendant = base_pendant or bail-maxh=0.5;
#solid pendant = bail;

solid rest = boxout and not pendant;

tlo rest -transparent -col=[0,0,1];#air
tlo pendant -col=[1,0.25,0.25];#ring -mur=1 -sig=4.03E+07