algebraic3d


solid Domain = orthobrick(-1000,-1000,-1000;1000,1000,1000);



#Create the body
solid bodyblock = orthobrick(0,-8.75,-95;148,8.75,6);

#######################################################
#Cut out the bits for the rail
solid Top_Rail_Cut1 = orthobrick(96,-10,0;150,10,7);
solid Top_Rail_Cut2 = orthobrick(30,-10,0;43,10,7)
    and plane(39,0,0;98.16,0,-19.08);
solid Top_Rail_Cut3 = orthobrick(-1,-9,0;100,-7,3)
    or orthobrick(-1,7,0;100,9,3);
solid Top_Rail_Cut4 = orthobrick(-1,-4,-9;40,4,10);


solid Rail_Cut = Top_Rail_Cut1 or Top_Rail_Cut2 or Top_Rail_Cut3 or Top_Rail_Cut4;



#######################################################
#Cut out the top bits from the body
solid Barrel_Cut1 = cylinder(96,0,-3.5;150,0,-3.5;7.5)
    and plane(96,0,0;-1,0,0)
	and plane(150,0,0;1,0,0);

solid Barrel_Cut2 = cylinder(78,0,-3.5;150,0,-3.5;6.5)
    and plane(78,0,0;-1,0,0)
	and plane(150,0,0;1,0,0);

solid Barrel_Cut3 = orthobrick(78,-5,-3.5;150,5,20);

solid Barrel_Cut4 = cylinder(65,0,10.4;150,0,10.4;7.5)
    and plane(65,0,0;-1,0,0)
	and plane(150,0,0;1,0,0);

solid Barrel_Cut5 = orthobrick(78,-2.5,-13;100,2.5,0)
    and cylinder(78,0,-3.5;150,0,-3.5;7.5);


solid Barrel_Cut =  Barrel_Cut4 or Barrel_Cut1 or Barrel_Cut2 or Barrel_Cut3 or Barrel_Cut5;



#######################################################
#Cut out the insides for the barrel
solid Front_Mag = cylinder(63.8008,0,0;44.7208,0,-98.16;6.5)
    or plane(63.8008,0,0;98.16,0,-19.08);

solid Mag_Cut1a = plane(33,0,0;-98.16,0,19.08)
    and plane(70.3008,0,0;98.16,0,-19.08)
	and plane(0,-7.5,0;0,-1,0)
	and plane(0,7.5,0;0,1,0)
	and plane(0,0,-20;0,0,1)
	and plane(0,0,-60;0,0,-1);
solid Mag_Cut1 = Mag_Cut1a and Front_Mag;

solid Mag_Cut2a = plane(33,0,0;-98.16,0,19.08)
    and plane(70.3008,0,0;98.16,0,-19.08)
	and plane(0,-6.5,0;0,-1,0)
	and plane(0,6.5,0;0,1,0)
	and plane(0,0,20;0,0,1)
	and plane(0,0,-21;0,0,-1);
solid Mag_Cut2 = Mag_Cut2a and Front_Mag;

solid Mag_Cut3a = plane(33,0,0;-98.16,0,19.08)
    and plane(70.3008,0,0;98.16,0,-19.08)
	and plane(0,-6.5,0;0,-1,0)
	and plane(0,6.5,0;0,1,0)
	and plane(0,0,-59;0,0,1)
	and plane(0,0,-100;0,0,-1);
solid Mag_Cut3 = Mag_Cut3a and Front_Mag;

solid Mag_Cut4 = cylinder(22.6728,-7.5,-53.1288;22.6728,7.5,-53.1288;7)
    and plane(0,-7.5,0;0,-1,0)
	and plane(0,7.5,0;0,1,0)
	and plane(0,0,-59.5;0,0,-1)
	and plane(26.2,0,0;-98.16,0,19.08);

solid Mag_Cut5 = plane(26.2,0,0;-98.16,0,19.08)
    and plane(50.2,0,0;98.16,0,-19.08)
	and plane(0,-7.5,0;0,-1,0)
	and plane(0,7.5,0;0,1,0)
	and plane(0,0,-22;0,0,1)
	and plane(0,0,-52.7;0,0,-1);

solid Mag_Cut6 = cylinder(18.202,-4.5,-76.1288;18.202,4.5,-76.1288;7)
    and plane(0,-4.5,0;0,-1,0)
	and plane(0,4.5,0;0,1,0)
	and plane(26.2,0,0;-98.16,0,19.08);

solid Mag_Cut7 = plane(26.2,0,0;-98.16,0,19.08)
    and plane(50.2,0,0;98.16,0,-19.08)
	and plane(0,-4.5,0;0,-1,0)
	and plane(0,4.5,0;0,1,0)
	and plane(0,0,-22;0,0,1)
	and plane(0,0,-76;0,0,-1);

solid Mag_Cut8 = orthobrick(-1,-4,-21;50,4,0)
    and not cylinder(0,-4,-30.5;0,4,-30.5;22);
	
solid Mag_Cut9 = orthobrick(-1,-7.5,-41;50,7.5,-20)
    and not cylinder(0,-4,-30.5;0,4,-30.5;22);


solid Mag_Cut = Mag_Cut1 or Mag_Cut2 or Mag_Cut3 or Mag_Cut4 or Mag_Cut5 or Mag_Cut6 or Mag_Cut7 or Mag_Cut8 or Mag_Cut9;




#######################################################
#Cut round the body of the gun
solid Outer_Cut1 = plane(25,0,0;98.16,0,-19.08)
    and cylinder(-2,-4,-33.5;-2,4,-33.5;21);

solid Outer_Cut2 = plane(25,0,0;98.16,0,-19.08)
    and plane(0,0,-33.5;0,0,1);

solid Outer_Cut3 = plane(71.14,0,0;-98.16,0,19.08)
    and cylinder(76.14,-10,-57;76.14,10,-57;16);
	
solid Outer_Cut4 = plane(71.14,0,0;-98.16,0,19.08)
    and plane(0,0,-55;0,0,1);

solid Outer_Cut5 = plane(0,0,-41.5;0,0,1)
    and plane(76.14,0,0;-1,0,0);

solid Outer_Cut6 = orthobrick(96,-10,-42;150,10,-30)
   and not cylinder(96,-10,-26.6;96,10,-26.6;15);
  
solid Outer_Cut7 = cylinder(122.1,-10,-22;122.1,10,-22;10);

solid Outer_Cut8 = plane(0,0,-20;0,0,1)
    and plane(117.64,0,0;-98.16,0,24.08);

solid Outer_Cut9 = orthobrick(122.1,-10,-32.1;150,10,-12.1);

solid Outer_Cut10 = orthobrick(-10,-10,-20;20,10,10)
   and not cylinder(25,-10,0;25,10,0;25);

solid Body_Cut = Outer_Cut1 or Outer_Cut2 or Outer_Cut3 or Outer_Cut4 or Outer_Cut5 or Outer_Cut6 or Outer_Cut7 or Outer_Cut8 or Outer_Cut9 or Outer_Cut10;




#######################################################
#Make a cut for the trigger hole
solid trigger_hole1 = cylinder(96,-10,-25;96,10,-25;13)
    or cylinder(85,-10,-25;85,10,-25;13)
	or orthobrick(85,-10,-37.9;96,10,-12.1);

solid trigger_side1 = plane(76.14,0,-31;-1,0,0)
    and plane(85,-10,-25;0,0,1)
	and plane(0,-4.5,0;0,1,0);

solid trigger_side2 = plane(76.14,0,-31;-1,0,0)
    and plane(85,-10,-25;0,0,1)
	and plane(0,4.5,0;0,-1,0);

solid trigger_side3 = plane(90,0,-12;-1,0,0)
    and plane(0,-4.5,0;0,1,0)
	and plane(0,0,-12.1;0,0,1);

solid trigger_side4 = plane(90,0,-12;-1,0,0)
    and plane(0,4.5,0;0,-1,0)
	and plane(0,0,-12.1;0,0,1);

solid trigger_hole = trigger_hole1 or trigger_side1 or trigger_side2 or trigger_side3 or trigger_side4;




#######################################################
#Cut out the pannel for the side
solid Side_Cut1 = plane(36,0,0;-98.16,0,19.08)
    and plane(63.3008,0,0;98.16,0,-19.08)
	and plane(0,-10,0;0,-1,0)
	and plane(0,10,0;0,1,0)
	and plane(0,0,-24;0,0,1)
	and plane(0,0,-79;0,0,-1);

solid Side_Cut2 = cylinder(36.22,-10,-23.9;36.22,10,-23.9;5)
    or cylinder(25.60,-10,-79.1;25.60,10,-79.1;5)
	or cylinder(53.67,-10,-23.9;53.67,10,-23.9;5)
	or cylinder(42.8808,-10,-79.1;42.8808,10,-79.1;5);

solid Side_Cut3 = plane(63.3008,0,0;98.16,0,-19.08)
    and plane(0,0,-79.75;0,0,-1)
	and cylinder(44.6808,-10,-79.1;44.6808,10,-79.1;5);
	
solid Side_Cut4 = plane(36,0,0;-98.16,0,19.08)
    and plane(0,0,-23.25;0,0,1)
	and cylinder(34.22,-10,-23.9;34.22,10,-23.9;5);

solid Side_Cut5 = orthobrick(36.22,-10,-25;53.67,10,-19)
    or orthobrick(25.6,-10,-84;42.8808,10,-78);
	
solid Side_Cut = Side_Cut1 or Side_Cut2 or Side_Cut3 or Side_Cut5;





#######################################################
#Make the chamfers round the body
solid Front_Chamfer1 = plane(63.8008,0,0;-98.16,0,19.08)
    and plane(76.14,0,-56;19.08,0,98.16)
    and not cylinder(61.0008,0,0;42.0208,0,-98.16;10);

solid Front_Chamfer2 = cylinder(76.14,-10,-57;76.14,10,-57;21)
    and plane(76.14,0,-58;-19.08,0,-98.16)
	and plane(76.44,0,0;1,0,0)
	and not torus(76.14,0,-57; 0,1,0;26;10.1);

solid Front_Chamfer3 = plane(76.14,-10,-57;-1,0,0)
    and plane(76.14,0,-31;0,0,1)
    and not cylinder(76.14,0,-31.2;96.14,0,-31.2;10);

solid Front_Chamfer4 = plane(96,0,0;-1,0,0)
   and plane(96,0,-25;19.08,0,98.16)
   and not torus(96,0,-25;0,1,0;6;10);

solid Front_Chamfer5 = cylinder(85,-10,-25;85,10,-25;18)
   and not torus(85,0,-25;0,1,0;23;10.1);

solid Front_Chamfer6 = plane(68.3,-8.75,-35.5;-19.08,0,48.16)
   and plane(68.3,-8.75,-35.5;-19.08,0,-8.16)
   and plane(68.3,-4.5,-35.5;0,1,0)
   and cylinder(68.3008,-10,-35.5;68.3008,10,-35.5;12)
   and not ellipsoid(67.3008,0,-35.5;8,0,0;0,0,8;0,9.5,0);

solid Front_Chamfer7 = plane(68.3,-8.75,-35.5;-19.08,0,48.16)
   and plane(68.3,-8.75,-35.5;-19.08,0,-8.16)
   and plane(68.3,4.5,-35.5;0,-1,0)
   and cylinder(68.3008,-10,-35.5;68.3008,10,-35.5;12)
   and not ellipsoid(67.3008,0,-35.5;8,0,0;0,0,8;0,9.5,0);

solid Front_Chamfer8 = plane(68.3,-4.5,-35.5;0,1,0)
   and plane(86,0,0;-1,0,0)
   and plane(0,0,-3;0,0,1)
   and not cylinder(0,0,-3;150,0,-3;10);
  
solid Front_Chamfer9 = plane(68.3,4.5,-35.5;0,-1,0)
   and plane(86,0,0;-1,0,0)
   and plane(0,0,-3;0,0,1)
   and not cylinder(0,0,-3;150,0,-3;10);

solid Front_Chamfer10 = cylinder(122.1,-10,-22;122.1,10,-22;20)
    and not torus(122.1,0,-22;0,1,0;20;10.1);

solid Front_Chamfer11 = plane(0,0,-20;0,0,1)
    and plane(112.64,0,0;-98.16,0,24.08)
	and not cylinder(107.54,0,0;83.46,0,-98.16;10);

solid Front_Chamfer12 = orthobrick(122.1,-9,-40;150,9,0)
    and not cylinder(122.1,0,-1;150,0,-1;11.5);

solid Back_Chamfer1 = cylinder(-2,-4,-33.5;-2,4,-33.5;26)
    and plane(0,0,-42.5;0,0,-1)
    and not torus(-4,0,-31.5;0,1,0;30;10);

solid Back_Chamfer2 = plane(36,0,0;98.16,0,-19.08)
    and plane(0,0,-32;0,0,1)
	and not cylinder(45,0,0;25.92,0,-98.16;20);

solid Back_Chamfer3 = orthobrick(0,-10,-100;25,10,-80)
    and not cylinder(23.03,-10,-85;23.503,10,-85;15);

solid Back_Chamfer4 = plane(23.03,0,-85;1,0,0)
    and plane(23.03,0,-85;19.08,0,98.16)
	and not ellipsoid(23.03,0,-85;15,0,0;0,0,15;0,17,0);



solid chamfers = Front_Chamfer1 or Front_Chamfer2 or Front_Chamfer3 or Front_Chamfer4 or Front_Chamfer5 or Front_Chamfer6 or Front_Chamfer7 or Front_Chamfer8 or Front_Chamfer9 or Front_Chamfer10 or Front_Chamfer11 or Front_Chamfer12 or Back_Chamfer1 or Back_Chamfer2 or Back_Chamfer3 or Back_Chamfer4;




#######################################################
#Cut out the other little holes
solid safety_hole = cylinder(67.3008,-10,-35;67.3008,10,-35;2);

solid trigger_cut = plane(60,-2.5,-38;-1,0,0)
    and plane(70,-2.5,-38;0,-1,0)
	and plane(70,2.5,-38;0,1,0)
    and plane(70,-2.5,-38;0,0,-1)
	and plane(86,0,0;1,0,0)
	and plane(85,0,-13;19.08,0,98.16);

solid hole1 = cylinder(91,-10,-6;91,10,-6;2);
solid hole2 = cylinder(63,-10,-3;63,10,-3;3.5)
    and plane(0,0,0;0,-1,0);

solid hole3 = cylinder(17.202,-10,-69;17.202,10,-69;1);

solid hole4 = cylinder(21.202,-10,-93;21.202,10,-93;1)
    and plane(0,0,0;0,-1,0);

solid hole5 = cylinder(12.202,-10,-86;12.202,10,-86;1)
    and plane(0,-6,0;0,-1,0);


solid holes = safety_hole or trigger_cut or hole1 or hole2 or hole3 or hole4 or hole5;



solid body = bodyblock and not Rail_Cut and not Mag_Cut and not Barrel_Cut and not Body_Cut and not trigger_hole and not Side_Cut and not chamfers and not holes;



solid rest = Domain and not body;


tlo rest -transparent -col=[0,0,0];#air
tlo body -col=[1,0,0];#steel -mur=5 -sig=5E+05

















