algebraic3d

solid boxout = orthobrick (-1000, -1000, -1000; 1000, 1000, 1000);
solid boxin = orthobrick (-100, -100, -100; 100, 100, 100);


#Create the main piece to cut away from
solid maincylinder = cylinder ( 0, 0, -0.85; 0, 0, 0.85; 10.98 )
	and plane (0, 0, -0.85; 0, 0, -1)
	and plane (0, 0, 0.85; 0, 0, 1)-maxh=0.8;



#each of the sides
#cut1
solid cut1 = cylinder ( 10.98, 0, -0.85; 10.98, 0, 0.85; 21.4 );

#cut2
solid cut2 = cylinder ( 6.84, 8.58, -0.85; 6.84, 8.58, 0.85; 21.4 );

#cut3
solid cut3 = cylinder ( -2.44, 10.7, -0.85; -2.44, 10.7, 0.85; 21.4 );

#cut4
solid cut4 = cylinder ( -9.89, 4.76, -0.85; -9.89, 4.76, 0.85; 21.4 );

#cut5
solid cut5 = cylinder ( -9.89, -4.76, -0.85; -9.89, -4.76, 0.85; 21.4 );

#cut6
solid cut6 = cylinder ( -2.44, -10.7, -0.85; -2.44, -10.7, 0.85; 21.4 );

#cut7
solid cut7 = cylinder ( 6.84, -8.58, -0.85; 6.84, -8.58, 0.85; 21.4 );



solid coin = maincylinder and cut1
		and cut2 and cut3
		and cut4 and cut5 
		and cut6 and cut7;



solid rest = boxout and not coin;

tlo rest -transparent -col=[0,0,1];#air
tlo coin-col=[1,0.25,0.25];#coin -mur=1 -sig=5.26E+06
