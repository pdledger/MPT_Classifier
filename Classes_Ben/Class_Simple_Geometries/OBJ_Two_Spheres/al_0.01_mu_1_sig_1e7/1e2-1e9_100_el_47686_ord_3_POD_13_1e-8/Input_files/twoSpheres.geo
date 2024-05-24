algebraic3d
#
# Example with two sub-domains:  was 10
#
solid sphout = sphere (0, 0, 0; 100) -bco=1;

solid sphin2 = sphere (-0.5, 0, 0; 1)
	and plane (0, 0, 0; 1, 0, 0) -bco=2;

solid sphinin2 = sphere (0.5, 0, 0; 1)
	and plane (0, 0, 0; -1, 0, 0) -bco=2;

solid hat=sphinin2 and not sphin2;

solid object = hat or sphin2  -maxh=0.1;

solid rest=sphout and not hat and not sphin2;

tlo rest -transparent -col=[0,0,1];#air
tlo object -col=[1,0,0];#mat -mur=1 -sig=1E+07
