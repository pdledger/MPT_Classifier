algebraic3d

solid boxout = orthobrick (-1000, -1000, -1000; 1000, 1000, 1000);

solid head = orthobrick(-50,-25,-22.5;50,25,22.5)
	and not orthobrick(-15,-10,-40;15,10,40);




solid hammer = head-maxh=3;
solid rest = boxout and not hammer;

tlo rest -transparent -col=[0,0,1];#air
tlo hammer -col=[1,0.25,0.25];#ring -mur=1 -sig=4.03E+07