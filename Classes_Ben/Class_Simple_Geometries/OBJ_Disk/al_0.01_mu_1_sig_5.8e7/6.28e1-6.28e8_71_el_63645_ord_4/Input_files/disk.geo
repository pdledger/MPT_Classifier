algebraic3d
#
# Example with two sub-domains
#
solid boxout = orthobrick (-100, -100, -100; 100, 100, 100);
solid cylin = cylinder ( 0, 0, -0.11; 0, 0, 0.11; 1.53 )
	and plane (0, 0, -0.11; 0, 0, -0.11)
	and plane (0, 0, 0.11; 0, 0, 0.11) -maxh=0.06;

solid rest = boxout and not cylin;

tlo rest -transparent -col=[0,0,1];#air
tlo cylin-col=[1,0,0];#cylinder -mur=1 -sig=5.8E+07