algebraic3d

solid boxout = orthobrick (-1000, -1000, -1000; 1000, 1000, 1000);

solid inner = cylinder ( 0, 0, -1.4; 0, 0, 1.4; 7.6 )
	and plane (0, 0, -1.4; 0, 0, -1)
	and plane (0, 0, 1.4; 0, 0, 1) -maxh=1;
	
solid outercyl = cylinder ( 0, 0, -1.4; 0, 0, 1.4; 11.665 )
	and plane (0, 0, -1.4; 0, 0, -1)
	and plane (0, 0, 1.4; 0, 0, 1) -maxh=1;
	
solid cut1 = plane (11.31,0,0;-11.31,0,0);
solid cut2 = plane (9.80,5.66,0;-9.80,-5.66,0);
solid cut3 = plane (5.66,9.80,0;-5.66,-9.80,0);
solid cut4 = plane (0,11.31,0;0,-11.31,0);
solid cut5 = plane (-9.80,5.66,0;9.80,-5.66,0);
solid cut6 = plane (-5.66,9.80,0;5.66,-9.80,0);

solid cut7 = plane (-11.31,0,0;11.31,0,0);
solid cut8 = plane (-9.80,-5.66,0;9.80,5.66,0);
solid cut9 = plane (-5.66,-9.80,0;5.66,9.80,0);
solid cut10 = plane (0,-11.31,0;0,11.31,0);
solid cut11 = plane (9.80,-5.66,0;-9.80,5.66,0);
solid cut12 = plane (5.66,-9.80,0;-5.66,9.80,0);

solid outer = outercyl and not inner
		and not cut1 and not cut2
		and not cut3 and not cut4
		and not cut5 and not cut6
		and not cut7 and not cut8
		and not cut9 and not cut10
		and not cut11 and not cut12;

solid rest = boxout and not inner and not outer;

tlo rest -transparent -col=[0,0,1];#air
tlo inner -transparent -col =[0,1,0];#inner -mur=1 -sig=1.63E+07
tlo outer -col=[1,0,0];#outer -mur=1.15 -sig=5.26E+06
