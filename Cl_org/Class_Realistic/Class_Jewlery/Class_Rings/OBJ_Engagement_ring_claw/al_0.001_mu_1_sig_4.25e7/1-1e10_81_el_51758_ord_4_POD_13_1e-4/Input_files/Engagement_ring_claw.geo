algebraic3d

solid boxout = orthobrick (-1000, -1000, -1000; 1000, 1000, 1000);


#rings should vary from 14-22.6mm diameter (size 3-13.5)

#define the shape of the ring
solid base_ring = torus(0,0,0;0,0,1;6.2;3)
	and torus(0,0,0;0,0,1;12;4)
	and torus(0,0,0;0,0,1;8.5;1);

solid claw1 = cylinder(8,0,0;11,1.5,1.5;0.25)
	and plane(8.5,0,0;-1,0,0)
	and plane(11,1.5,1.5;3,1.5,1.5);

solid claw2 = cylinder(8,0,0;11,1.5,-1.5;0.25)
	and plane(8.5,0,0;-1,0,0)
	and plane(11,1.5,-1.5;3,1.5,-1.5);

solid claw3 = cylinder(8,0,0;11,-1.5,1.5;0.25)
	and plane(8.5,0,0;-1,0,0)
	and plane(11,-1.5,1.5;3,-1.5,1.5);

solid claw4 = cylinder(8,0,0;11,-1.5,-1.5;0.25)
	and plane(8.5,0,0;-1,0,0)
	and plane(11,-1.5,-1.5;3,-1.5,-1.5);

solid claws = claw1 or claw2 or claw3 or claw4 -maxh=0.1;


solid ring = base_ring or claws;


solid rest = boxout and not ring;

tlo rest -transparent -col=[0,0,1];#air
tlo ring -col=[1,0.25,0.25];#ring -mur=1 -sig=4.03E+07