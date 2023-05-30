algebraic3d

solid boxout = orthobrick (-1000, -1000, -1000; 1000, 1000, 1000);


solid pendant = cylinder(4.5,0,0;4.5,0,1;5)
	and plane(0,0,0.5;0,0,1)
	and plane(0,0,-0.5;0,0,-1)
	or torus(-0.75,0,0;0,0,1;1;0.35)-maxh=0.5;


solid rest = boxout and not pendant;

tlo rest -transparent -col=[0,0,1];#air
tlo pendant -col=[1,0.25,0.25];#ring -mur=1 -sig=4.03E+07