algebraic3d

solid boxout = orthobrick (-1000, -1000, -1000; 1000, 1000, 1000);


#Create the buckle
solid buckle = orthobrick(-40,-30,-5;40,30,5)
	and not orthobrick(-32,-22,-6;32,22,6);

solid spike = orthobrick(-35,-2.5,0;35,2.5,5);





solid full_buckle = buckle or spike-maxh=2.5;

solid rest = boxout and not full_buckle;

tlo rest -transparent -col=[0,0,1];#air
tlo full_buckle -col=[1,0.25,0.25];#shank -mur=1 -sig=4.03E+07