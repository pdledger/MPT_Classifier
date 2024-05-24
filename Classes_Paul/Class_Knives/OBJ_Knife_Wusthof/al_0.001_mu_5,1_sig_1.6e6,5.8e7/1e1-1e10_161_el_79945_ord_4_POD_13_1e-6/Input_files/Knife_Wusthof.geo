algebraic3d


solid Domain = orthobrick(-1000,-1000,-1000;1000,1000,1000);


#Create the blade
solid blade = ellipticcylinder(0,0,0;200,0,0;0,31.5,0)
    and ellipticcylinder(0,-13.0,0;222.0,0,0;0,31.5,0)
    and plane(0,0,0;-1,0,0)
    and plane(0,18.5,-1.0;0,-0.8,-50.0)
    and plane(0,18.5,1.0;0,-0.8,50.0);



#Create the bolster
solid heelmain = orthobrick(-30,-31.5,-8.25;0,18.5,8.25);
solid cutfrontleft = cylinder(0,18.5,-16.5;0,-31.5,-15.7;16.3)
    and plane(0,18.5,-1.0;0,0.8,50.0);
solid cutfrontright = cylinder(0,18.5,16.5;0,-31.5,15.7;16.3)
    and plane(0,18.5,1.0;0,0.8,-50.0);
solid cutbackleft = cylinder(-30,18.5,-8.25;-30,-31.5,-8.25;6.5);
solid cutbackright = cylinder(-30,18.5,8.25;-30,-31.5,8.25;6.5);
solid cutunder1 = cylinder(-27.0,-23.5,-8.25;-27.0,-23.5,8.25;12.4);
solid cutunder2 = plane(-15.0,-23.5,0;1,0,0)
    and plane(-15.0,-23.5,0;0,1,0);
solid cutunder3 = plane(-27.0,-11.5,0;0,1,0)
    and plane(-27.0,-11.5,0;1,0,0);
solid heelrounder = ellipticcylinder(0,-3.5,0;0,-29.0,0;0,0,7.5);

solid cutunder = cutunder1 or cutunder2 or cutunder3;
solid heelcuts = cutfrontleft or cutfrontright or cutbackleft or cutbackright or cutunder1 or cutunder;

solid heel = heelmain and heelrounder and not heelcuts and not heelcuts;



#Create the tang
solid tangblock = orthobrick(-150,-11.5,-1.75;-30,18.5,1.75);

#Add the rivets
solid rivet1 = cylinder(-47.5,3.5,-10.0;-47.5,3.5,10.0;3.5)
    and plane(0,0,-10.0;0,0,-1)
    and plane(0,0,10.0;0,0,1);

solid rivet2 = cylinder(-90.0,3.5,-10.0;-90.0,3.5,10.0;3.5)
    and plane(0,0,-10.0;0,0,-1)
    and plane(0,0,10.0;0,0,1);

solid rivet3 = cylinder(-132.0,3.5,-10.0;-132.0,3.5,10.0;3.5)
    and plane(0,0,-10.0;0,0,-1)
    and plane(0,0,10.0;0,0,1);

solid rivets = rivet1 or rivet2 or rivet3-maxh=1;

solid tang = tangblock and not rivets;



#Create the butt of the knife
solid butt = ellipticcylinder(-150,-4.0,0;18.0,0,0;0,22.5,0)
    and plane(0,0,-1.75;0,0,-1)
    and plane(0,0,1.75;0,0,1)
    and plane(-133.0,0,0;1,0,0)
    and not orthobrick(-150,-11.5,-2.1;-105.0,18.5,2.1)
    and not cylinder(-133.0,-34.0,-1;-133.0,-34.0,1;23.6);



solid rest = Domain and not blade and not heel and not tang and not rivets and not butt;

tlo rest -transparent -col=[0,0,1];#air
tlo blade -col=[1,0,0];#Blade -mur=5 -sig=1.6E+06
tlo heel -col=[0,1,0];#Bolster -mur=5 -sig=1.6E+06
tlo tang -col=[0,0,1];#Tang -mur=5 -sig=1.6E+06
tlo rivets -col=[1,1,0];#Rivets -mur=1 -sig=5.8E+07
tlo butt -col=[1,0,1];#Butt -mur=5 -sig=1.6E+06

