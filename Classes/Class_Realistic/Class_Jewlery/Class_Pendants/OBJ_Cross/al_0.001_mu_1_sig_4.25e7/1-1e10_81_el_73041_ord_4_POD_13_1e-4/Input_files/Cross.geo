algebraic3d

solid boxout = orthobrick (-1000, -1000, -1000; 1000, 1000, 1000);


solid pendant = orthobrick(0,-1,-0.75;19,1,0.75)
	or orthobrick(5,-6,-0.75;7,6,0.75)
	or torus(-0.5,0,0;0,0,1;1;0.35)-maxh=0.5;


solid rest = boxout and not pendant;

tlo rest -transparent -col=[0,0,1];#air
tlo pendant -col=[1,0.25,0.25];#ring -mur=1 -sig=4.03E+07