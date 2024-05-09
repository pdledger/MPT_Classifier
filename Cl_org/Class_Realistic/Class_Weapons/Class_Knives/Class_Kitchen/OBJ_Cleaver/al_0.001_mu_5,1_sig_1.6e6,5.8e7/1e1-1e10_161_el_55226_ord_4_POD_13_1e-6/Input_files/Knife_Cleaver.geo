algebraic3d


solid Domain = orthobrick(-1000,-1000,-1000;1000,1000,1000);


#Create the blade
solid bladeblank = orthobrick(0,-66.4,-1.5;200,45.6,1.5)
    and ellipticcylinder(100.0,21.6,0;300.0,0,0;0,80.0,0)
    and plane(0,-34.4,-1.5;0,-3.84,-80.0)
    and plane(0,-34.4,1.5;0,-3.84,80.0)
    and not ellipticcylinder(0,61.6,-1.5;260.0,0,0;0,32.0,0);

solid bladetip = plane(189.0,0,0;-1,0,0)
    and plane(0,29.6,0;0,-1,0)
    and not cylinder(191.0,31.2,-1;191.0,31.2,1;9.6);

solid blade = bladeblank and not bladetip-maxh=10;



#Create the bolster
solid heelblank = orthobrick(-25,-50.4,-1.75;0,29.6,1.75);

solid heelcut1 = plane(0,-5.4,0;0,1,0)
    and plane(-12.5,0,0;1,0,0);

solid heelcut2 = plane(0,-17.9,0;0,1,0);

solid heelcut3 = cylinder(-12.2,-17.5,-1;-12.2,-17.5,1;12.5);

solid heelcut = heelcut1 or heelcut2 or heelcut3;

solid heeltaper = plane(0,0,1.5;-0.25,0,-25)
    or plane(0,0,-1.5;-0.25,0,25);
solid heel = heelblank and not heelcut and not heeltaper-maxh=10;


#Create the tang
solid tangblank = orthobrick(-145,-5.4,-1.75;-25,29.6,1.75)
    and ellipticcylinder(-85.0,29.6,0;84.0,0,0;0,28.0,0);

solid tangfront = orthobrick(-51.4,-5.4,-1.75;-25,29.6,1.75)
    and not cylinder(-49.0,-8.9,-1;-49.0,-8.9,1;13.3);

solid tangback = orthobrick(-145,-5.4,-1.75;-119.0,29.6,1.75)
    and not cylinder(-121.0,-8.9,-1;-121.0,-8.9,1;13.3);

solid tangend = orthobrick(-127.0,-5.4,-1.75;-25,29.6,1.75)
    or cylinder(-127.0,12.1,-1;-127.0,12.1,1;17.9);
#Add the rivets
solid rivet1 = cylinder(-40.0,15.6,-8.0;-40.0,15.6,8.0;3)
    and plane(0,0,-10.0;0,0,-1)
    and plane(0,0,10.0;0,0,1);

solid rivet2 = cylinder(-85.0,15.6,-8.0;-85.0,15.6,8.0;3)
    and plane(0,0,-10.0;0,0,-1)
    and plane(0,0,10.0;0,0,1);

solid rivet3 = cylinder(-130.0,15.6,-8.0;-130.0,15.6,8.0;3)
    and plane(0,0,-10.0;0,0,-1)
    and plane(0,0,10.0;0,0,1);

solid rivets = rivet1 or rivet2 or rivet3-maxh=1;

solid tangblock = tangblank or tangfront or tangback;
solid tang = tangblock and tangend and not rivets;




solid rest = Domain and not blade and not heel and not tang and not rivets;

tlo rest -transparent -col=[0,0,1];#air
tlo blade -col=[1,0,0];#Blade -mur=5 -sig=1.6E+06
tlo heel -col=[0,1,0];#Bolster -mur=5 -sig=1.6E+06
tlo tang -col=[0,0,1];#Tang -mur=5 -sig=1.6E+06
tlo rivets -col=[1,1,0];#Rivets -mur=1 -sig=5.8E+07

