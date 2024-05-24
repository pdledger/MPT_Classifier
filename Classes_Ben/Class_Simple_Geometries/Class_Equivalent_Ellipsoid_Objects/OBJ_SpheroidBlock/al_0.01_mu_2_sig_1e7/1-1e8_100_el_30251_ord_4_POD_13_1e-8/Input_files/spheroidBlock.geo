algebraic3d
#
# Example with two sub-domains: 
#
solid ellpout = ellipsoid (0, 0, 0; 100, 0, 0; 0, 100, 0; 0, 0, 100) -bco=1;
solid ellpin = ellipsoid (0, 0, 0; 0.6877, 0, 0; 0, 1.2366, 0; 0, 0, 1.7047) -bco=2;

solid rest = ellpout and not ellpin;
solid object= ellpin  -maxh=0.2;

tlo rest -transparent -col=[0,0,1];#air
tlo object -col=[1,0,0];#spheroid -mur=2 -sig=1E+07

