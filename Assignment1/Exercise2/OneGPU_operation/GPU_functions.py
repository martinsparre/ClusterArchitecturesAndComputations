import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
import sys



   
def get_kernel(ctx, NP,GRAV_CONST,TIMESTEP):
    """
    This returns a compiled kernels
    """

        
    kernel_src_template = """

    #define GRAV_CONST %(GRAV_CONST)f
    #define TIMESTEP %(TIMESTEP)f
    #define N_PARTICLES %(NP)i

   
    __kernel void CalcF(__global float *X,
                      __global float *Y,
                      __global float *Z,                   
                      __global float *M,
                      __global float *dvx,
                      __global float *dvy,
                      __global float *dvz                       
                      ) {
                       
       int i = get_global_id(0);
       int j;
       float tmp;

       for (j=0;j<N_PARTICLES;j++){
            tmp = - GRAV_CONST * M[j]  / pow(pow(X[i]-X[j],2)+pow(Y[i]-Y[j],2)+pow(Z[i]-Z[j],2)+((float)0.0000001),(float)1.5)*TIMESTEP;              
            dvx[j] += tmp * (X[j]-X[i]);
            dvy[j] += tmp * (Y[j]-Y[i]);
            dvz[j] += tmp * (Z[j]-Z[i]);
            }
            
            
        //SYNCHRONIZE.....
            
        //X=X+V*DT
        //V=V+DV
     
        
    }
    """
                                 #
    kernel_src = kernel_src_template % {
                     'GRAV_CONST' : GRAV_CONST,
                     'TIMESTEP' : TIMESTEP,
                     'NP' : NP
                 }
    print kernel_src
                 
    kernel = cl.Program(ctx, kernel_src).build()
                                                                                           
    return kernel

def CalcF(ctx,queue,x,y,z,m,GRAV_CONST,TIMESTEP):
    # Define dimensions
    xdim = m.shape[0]


    kernel = get_kernel(ctx, len(x),GRAV_CONST,TIMESTEP)
    
    # Move data to the GPU
    gpu_x = cl_array.to_device(queue, x)
    gpu_y = cl_array.to_device(queue, y)
    gpu_z = cl_array.to_device(queue, z)
    gpu_m = cl_array.to_device(queue, m)
  
    gpu_dvx = cl_array.zeros(queue, (xdim,), np.float32)
    gpu_dvy = cl_array.zeros(queue, (xdim,), np.float32)    
    gpu_dvz = cl_array.zeros(queue, (xdim,), np.float32)
    
    # Define grid shape (the same as the matrix dimensions)
    grid_shape = (100,)
    
    # Get group shape based on the matrix dimensions and the actual hardware
    group_shape = (10,)
    

    event = kernel.CalcF(queue, 
                       grid_shape, group_shape, 
                       gpu_x.data, 
                       gpu_y.data,
                       gpu_z.data,
                       gpu_m.data,
                       gpu_dvx.data,
                       gpu_dvy.data,
                       gpu_dvz.data)
                       
    event.wait()    
    dvx = gpu_dvx.get()
    dvy = gpu_dvx.get()
    dvz = gpu_dvx.get()
    queue.finish()

    return dvx,dvy,dvz
