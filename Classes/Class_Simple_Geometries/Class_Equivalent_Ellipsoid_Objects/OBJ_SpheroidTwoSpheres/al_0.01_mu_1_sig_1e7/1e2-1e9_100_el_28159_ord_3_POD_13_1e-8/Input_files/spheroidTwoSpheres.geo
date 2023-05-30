algebraic3d
#
# Example with two sub-domains: 
#
solid ellpout = ellipsoid (0, 0, 0; 100, 0, 0; 0, 100, 0; 0, 0, 100) -bco=1;
solid ellpin = ellipsoid (0, 0, 0; 1.0552, 0, 0; 0, 1.0552, 0; 0, 0, 1.5268) -bco=2;

solid rest = ellpout and not ellpin;
solid object= ellpin  -maxh=0.15;

tlo rest -transparent -col=[0,0,1];#air
tlo object -col=[1,0,0];#spheroid -mur=1 -sig=1E+7

