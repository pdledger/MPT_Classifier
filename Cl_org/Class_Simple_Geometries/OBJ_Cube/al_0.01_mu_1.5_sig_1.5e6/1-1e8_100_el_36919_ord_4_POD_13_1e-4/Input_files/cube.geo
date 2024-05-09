algebraic3d

solid boxout = orthobrick (-100, -100, -100; 100, 100, 100) -bco=1;

solid cb1 = orthobrick ( 0, 0, 0; 1, 1, 1) -bco=2;

solid rest = boxout and not cb1;

solid object= cb1  -maxh=0.1;

tlo rest -transparent -col=[0,0,1];#air
tlo object -col=[1,0,0];#mat1 -mur=1.5 -sig=1.5E+06


