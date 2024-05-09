algebraic3d

solid boxout = orthobrick (-1000, -1000, -1000; 1000, 1000, 1000);
solid boxin = orthobrick (-100, -100, -100; 100, 100, 100);


#Create the main piece to cut away from
solid maincylinder = cylinder ( 0, 0, -0.89; 0, 0, 0.89; 14.00 )
	and plane (0, 0, -0.89; 0, 0, -1)
	and plane (0, 0, 0.89; 0, 0, 1)-maxh=1;



#each of the sides
#cut1
solid cut1 = cylinder ( 14.00, 0, -0.89; 14.00, 0, 0.89; 27.25 );

#cut2
solid cut2 = cylinder ( 8.73, 10.95, -0.89; 8.73, 10.95, 0.89; 27.25 );

#cut3
solid cut3 = cylinder ( -3.11, 13.65, -0.89; -3.11, 13.65, 0.89; 27.25 );

#cut4
solid cut4 = cylinder ( -12.62, 6.07, -0.89; -12.62, 6.07, 0.89; 27.25 );

#cut5
solid cut5 = cylinder ( -12.62, -6.07, -0.89; -12.62, -6.07, 0.89; 27.25 );

#cut6
solid cut6 = cylinder ( -3.11, -13.65, -0.89; -3.11, -13.65, 0.89; 27.25 );

#cut7
solid cut7 = cylinder ( 8.73, -10.95, -0.89; 8.73, -10.95, 0.89; 27.25 );



solid coin = maincylinder and cut1
		and cut2 and cut3
		and cut4 and cut5 
		and cut6 and cut7;



solid rest = boxout and not coin;

tlo rest -transparent -col=[0,0,1];#air
tlo coin-col=[1,0.25,0.25];#coin -mur=1 -sig=2.91E+06
