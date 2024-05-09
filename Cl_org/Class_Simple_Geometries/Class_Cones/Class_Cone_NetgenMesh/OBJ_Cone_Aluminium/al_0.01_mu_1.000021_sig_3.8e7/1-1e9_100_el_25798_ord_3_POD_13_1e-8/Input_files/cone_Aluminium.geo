#
## A cone
#
algebraic3d

# Cone given by bottom circle and top circle
# and cut by planes:

solid sphout = sphere (0, 0, 0; 200) -bco=1;

solid cutcone = cone ( 0, 0, 0; 1.5; 3, 0, 0; 0.005 )
	and plane (0, 0, 0; -1, 0, 0)
	and plane (3, 0, 0; 1, 0, 0);



solid rest = sphout and not cutcone;

solid object= cutcone  -maxh=0.13;

tlo rest -transparent -col=[0,0,1];#air
tlo object -col=[1,0,0];#mat1 -mur=1.000021 -sig=38.0E+06


