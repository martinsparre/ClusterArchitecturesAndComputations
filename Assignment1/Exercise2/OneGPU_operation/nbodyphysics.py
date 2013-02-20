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
import numpy, time
import GPU_functions
import pyopencl as cl
# By using the solar-mass as the mass unit and years as the standard time-unit
# the gravitational constant becomes 1

G = 1.0
DebugMode = False
Timing = True

class Galaxy:
    def __init__(self, x_max, y_max, z_max,n,max_mass=40):
        if DebugMode == True:
            x = [-3.0,7.0]
            y = [0.0,0.0]
            z = [0.0,0.0]
            vx = [0.0,0.0]
            vy = [0.0,0.0]
            vz = [0.0,0.0]
            m = [17.0,13.0]

            self.x = numpy.array(x)
            self.y = numpy.array(y)
            self.z = numpy.array(z)        
            self.vx = numpy.array(vx)        
            self.vy = numpy.array(vy)        
            self.vz = numpy.array(vz)
            self.m = numpy.array(m)
            self.n = len(self.x)
        else:
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

        self.x = numpy.float32(self.x)
        self.y = numpy.float32(self.y)
        self.z = numpy.float32(self.z)        
        self.vx = numpy.float32(self.vx)        
        self.vy = numpy.float32(self.vy)        
        self.vz = numpy.float32(self.vz)
        self.m = numpy.float32(self.m)
        self.n = len(self.x)
        
        
        
def calc_force(Gal,dt):
    """Calculate forces between bodies
    F = ((G m_a m_b)/r^2)/((x_b-x_a)/r)
   """
    
    ctx = cl.create_some_context(0)#use device 0, the GPU
    queue = cl.CommandQueue(ctx)    

    if Timing:
        start = time.time()

    #Convention: dx[i,j] = x[i] - x[j]

    Gal.dvx,Gal.dvy,Gal.dvz = GPU_functions.CalcF(ctx,queue,Gal.x,Gal.y,Gal.z,Gal.m,1.0,1.0)
    if Timing:
        stop = time.time()    
        print 'Time for F_ij computation', stop-start    

#    
    if DebugMode==True:
#
        print 'Check that the force is attracting'
        print Gal.x, Gal.y, Gal.z
        print Gal.dvx, Gal.dvy, Gal.dvz
        print '----End check'        

    print 'Check that the force is attracting'
    print Gal.x[120]
    print Gal.dvx[120]
    print '----End check'  


def move(galaxy, dt):
    """Move the bodies
    first find forces and change velocity and then move positions
    """

        
    calc_force(galaxy)
    
    galaxy.x += galaxy.vx*dt
    galaxy.y += galaxy.vy*dt
    galaxy.z += galaxy.vz*dt
        
        
if __name__ == '__main__':
    g = Galaxy(32,32,32,1000)
    a = calc_force(g,1.0)
        