algebraic3d
#
# Example with two sub-domains:  was 10
#
solid sphout = sphere (0, 0, 0; 100) -bco=1;
solid torin1 = torus (2.5, 0, 0; 0,0,1;2; 1) -bco=2;
solid torin2 = torus (-2.5, 0, 0; 0,0,1;2; 1) -bco=2;

#solid torin = torin1;

solid rest = sphout and not torin1 and not torin2;

solid object = torin1 or torin2 -maxh=0.25;


tlo rest -transparent -col=[0,0,1];#air
tlo object -col=[1,0,0];#mat -mur=1 -sig=1E+07
