algebraic3d

solid boxout = orthobrick (-1000, -1000, -1000; 1000, 1000, 1000);
solid cylin = cylinder ( 0, 0, -0.825; 0, 0, 0.825; 10.15 )
        and plane (0, 0, -0.825; 0, 0, -1)
        and plane (0, 0, 0.825; 0, 0, 1) -maxh=0.7;

solid rest = boxout and not cylin;

tlo rest -transparent -col=[0,0,1];#air
tlo cylin-col=[1,0.25,0.25];#coin -mur=1 -sig=4.03E+07
