algebraic3d

solid boxout = orthobrick (-100, -100, -100; 100, 100, 100) -bco=1;

solid cb1 = orthobrick ( -0.5, -1, -1.5; 0.5, 1, 1.5) -bco=2;

solid rest = boxout and not cb1;

solid object= cb1  -maxh=0.15;

tlo rest -transparent -col=[0,0,1];#air
tlo object -col=[1,0,0];#mat1 -mur=2 -sig=1E+07


