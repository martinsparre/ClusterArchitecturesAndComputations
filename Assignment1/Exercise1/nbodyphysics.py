#!/usr/bin/python
# -*- coding: utf-8 -*-

"""NBody in N^2 complexity
Note that we are using only Newtonian forces and do not consider relativity
Neither do we consider collisions between stars
Thus some of our stars will accelerate to speeds beyond c
This is done to keep the simulation simple enough for teaching purposes

All the work is done in the calc_force, move and random_galaxy functions.
To vectorize the code these are the functions to transform.
"""
import numpy

# By using the solar-mass as the mass unit and years as the standard time-unit
# the gravitational constant becomes 1

G = 1.0



class Galaxy:
    def __init__(self, x_max, y_max, z_max,n,max_mass=40):
        self.x_max = x_max
        self.y_max = y_max
        self.z_max = z_max
        self.n = n
        self.max_mass = max_mass 
        self.m = 1.0 * numpy.array([ numpy.random.randint(1, self.max_mass) / (4 * numpy.pi ** 2) for i in xrange(n)])
        self.x = 1.0 * numpy.array([ numpy.random.randint(-x_max, x_max) for _ in xrange(n)])
        self.y = 1.0 * numpy.array([ numpy.random.randint(-y_max, y_max) for _ in xrange(n)])
        self.z = 1.0 * numpy.array([ numpy.random.randint(-z_max, z_max) for _ in xrange(n)])
        self.vx = self.vy = self.vz = 1.0 * numpy.zeros(n)


def calc_force(Gal, dt):
    """Calculate forces between bodies
    F = ((G m_a m_b)/r^2)/((x_b-x_a)/r)
   """
    
    dx = numpy.zeros((len(Gal.x),len(Gal.x)))
    dy = numpy.zeros((len(Gal.y),len(Gal.y)))
    dz = numpy.zeros((len(Gal.z),len(Gal.z)))    


    
    for i in range(len(Gal.x)):
        dx[i,] = Gal.x[i]-Gal.x
        dy[i,] = Gal.y[i]-Gal.y
        dz[i,] = Gal.z[i]-Gal.z#maybe create an array that has Gal.x in all its rows...
        
        
    print 'dx[i,j] = x[i] - x[j]'
    print Gal.x[12]-Gal.x[1], dx[12,1], dx[1,12]
    
    r2_ij = dx**2+dy**2+dz**2+0.0005**2
    r_ij = numpy.sqrt(r2_ij)
    m2_ij = numpy.outer(Gal.m,Gal.m)

    print m2_ij.shape
    print r2_ij.shape
    print r_ij.shape    
    Fij = G*m2_ij / r2_ij / r_ij * dt
    
    tmp = numpy.matrix(Fij)
    w=tmp * dx
    print w.shape

    dvx = numpy.zeros((len(Gal.x),len(Gal.x)))
    dvy = numpy.zeros((len(Gal.y),len(Gal.y)))
    dvz = numpy.zeros((len(Gal.z),len(Gal.z)))    
    
    for i in range(len(Gal.x)):
        print tmp.shape, dx[i,].shape
        dvx[i,] = numpy.dot(tmp , dx[i,])/ Gal.m
        dvy[i,] = numpy.dot(tmp , dy[i,])/ Gal.m
        dvz[i,] = numpy.dot(tmp , dz[i,])/ Gal.m
        
    
    
    return dx,dvx


def move(galaxy, dt):
    """Move the bodies
    first find forces and change velocity and then move positions
    """

    for i in galaxy:
        for j in galaxy:
            if i != j:
                calc_force(i, j, dt)

    for i in galaxy:
        i['x'] += i['vx']
        i['y'] += i['vy']
        i['z'] += i['vz']


def random_galaxy(
    x_max,
    y_max,
    z_max,
    n,
    ):
    """Generate a galaxy of random bodies"""

    max_mass = 40.0  # Best guess of maximum known star

    # We let all bodies stand still initially

    return [{
        'm': numpy.random.random() * numpy.random.randint(1, max_mass)
            / (4 * numpy.pi ** 2),
        'x': numpy.random.randint(-x_max, x_max),
        'y': numpy.random.randint(-y_max, y_max),
        'z': numpy.random.randint(-z_max, z_max),
        'vx': 0,
        'vy': 0,
        'vz': 0,
        } for _ in xrange(n)]


