algebraic3d

solid boxout = orthobrick (-100, -100, -100; 100, 100, 100) -bco=1;

solid cb1 = orthobrick (-0.5, -0.5, -0.5; 0.5, 0.5, 0.5) -bco=2;

solid rest = boxout and not cb1;

solid object= cb1  -maxh=0.06;

tlo rest -transparent -col=[0,0,1];#air
tlo object -col=[1,0,0];#mat1 -mur=1.0001 -sig=2.0E+06


