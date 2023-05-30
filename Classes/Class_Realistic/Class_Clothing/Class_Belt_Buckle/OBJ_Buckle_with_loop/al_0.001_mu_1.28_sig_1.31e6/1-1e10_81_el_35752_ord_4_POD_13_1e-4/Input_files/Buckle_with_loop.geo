algebraic3d

solid boxout = orthobrick (-1000, -1000, -1000; 1000, 1000, 1000);


#Create the buckle
solid buckle = orthobrick(-40,-23,-3.5;47.5,23,3.5)
	and not orthobrick(-30,-20,-6;15,20,6);

solid loop = orthobrick(21,-23,3.5;28.5,23,15)
	and not orthobrick(20.5,-20,3.5;30,20,12);

solid spike = orthobrick(-35,-2.5,0;20,2.5,5);





solid full_buckle = buckle or loop or spike-maxh=2.5;

solid rest = boxout and not full_buckle;

tlo rest -transparent -col=[0,0,1];#air
tlo full_buckle -col=[1,0.25,0.25];#shank -mur=1 -sig=4.03E+07