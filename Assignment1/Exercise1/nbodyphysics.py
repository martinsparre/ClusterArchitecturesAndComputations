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
    NStars = len(Gal.x)

    #Convention: dx[i,j] = x[i] - x[j]
    dx = numpy.repeat(Gal.x,NStars) - numpy.tile(Gal.x,NStars)
    dy = numpy.repeat(Gal.y,NStars) - numpy.tile(Gal.y,NStars) 
    dz = numpy.repeat(Gal.z,NStars) - numpy.tile(Gal.z,NStars)     
    dx.shape = (NStars,NStars)
    dy.shape = (NStars,NStars)
    dz.shape = (NStars,NStars)
    
    #r2_ij, r_ij and m2_ij are calculated:
    r2_ij = dx**2+dy**2+dz**2+0.0001**2
    r_ij = numpy.sqrt(r2_ij)
    m2_ij = numpy.outer(Gal.m,Gal.m)

    
    #Fij is used to compute dvx, dvy and dvz in an efficient way:
    F_ij = G*m2_ij / r2_ij / r_ij * dt
    
    dvx = numpy.sum(F_ij * dx,axis=1) / Gal.m
    dvy = numpy.sum(F_ij * dy,axis=1) / Gal.m
    dvz = numpy.sum(F_ij * dz,axis=1) / Gal.m    
    
    return None
 

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



        
        
if __name__ == '__main__':
    g = Galaxy(10,10,10,5)
    a = calc_force(g,1)
        