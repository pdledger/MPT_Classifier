algebraic3d

solid boxout = orthobrick (-1000, -1000, -1000; 1000, 1000, 1000);


#Create the buckle
solid buckle = orthobrick(-30,-22,-7;30,22,9.5)
	and not orthobrick(-45,-20,-5;45,20,7.5);

solid spike = cylinder(-22,0,4;-22,1,4;2.5)
	and plane(0,-22,0;0,-1,0)
	and plane(0,22,0;0,1,0);

solid top_spike = sphere(-22,21,4;4)
	and plane(0,21,0;0,-1,0);

solid bottom_spike = sphere(-22,-21,4;4)
	and plane(0,-21,0;0,1,0);
	





solid full_buckle = buckle or spike or top_spike or bottom_spike-maxh=2.5;

solid rest = boxout and not full_buckle;

tlo rest -transparent -col=[0,0,1];#air
tlo full_buckle -col=[1,0.25,0.25];#shank -mur=1 -sig=4.03E+07