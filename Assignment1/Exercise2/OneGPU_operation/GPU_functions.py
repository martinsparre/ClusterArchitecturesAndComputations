import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
import sys



   
def get_kernel(ctx, xdim,DIRECTION,GRAV_CONST,TIMESTEP):
    """
    This returns a compiled kernels
    """
    if DIRECTION not in ['X','Y','Z']:
        print '--- FATAL ERROR: DIRECTION should be X, Y, or Z.'
        sys.exit()
        
    kernel_src_template = """

    #define GRAV_CONST %(GRAV_CONST)f
    #define TIMESTEP %(TIMESTEP)f
    #define XDIM %(XDIM)i
    #define RESULT(y,x) result[(y*XDIM)+x]

   
    __kernel void CalcF(__global float *result,
                      __global float *X,
                      __global float *Y,
                      __global float *Z,
                      __global float *M
                      ) {
                       
       int x = get_global_id(0);
       int y = get_global_id(1);
   
       RESULT(y,x) = GRAV_CONST * M[x]  / pow(pow(X[x]-X[y],2)+pow(Y[x]-Y[y],2)+pow(Z[x]-Z[y],2)+((float)0.0000001),(float)1.5) * (%(DIRECTION)s[y]-%(DIRECTION)s[x])*TIMESTEP;
    }
    """
                                 #
    kernel_src = kernel_src_template % {
                     'XDIM' : xdim,
                     'GRAV_CONST' : GRAV_CONST,
                     'DIRECTION' : DIRECTION,
                     'TIMESTEP' : TIMESTEP
                 }
    print kernel_src
                 
    kernel = cl.Program(ctx, kernel_src).build()
                                                                                           
    return kernel

def CalcF(ctx,queue,x,y,z,m,DIRECTION,GRAV_CONST,TIMESTEP):

 
    # Define dimensions
    xdim = ydim = m.shape[0]


#    m2 = np.float32(m2)
#    r2 = np.float32(r2)
 
    # Get the compiled kernel
    kernel = get_kernel(ctx, xdim,DIRECTION,GRAV_CONST,TIMESTEP)
    
    # Move data to the GPU

    gpu_x = cl_array.to_device(queue, x)
    gpu_y = cl_array.to_device(queue, y)
    gpu_z = cl_array.to_device(queue, z)
    gpu_m = cl_array.to_device(queue, m)
  
    gpu_result = cl_array.zeros(queue, (ydim, xdim), np.float32)

    # Define grid shape (the same as the matrix dimensions)
    grid_shape = (ydim, xdim)
    
    # Get group shape based on the matrix dimensions and the actual hardware
    group_shape = (2,2)
    

    event = kernel.CalcF(queue, 
                       grid_shape, group_shape, 
                       gpu_result.data, 
                       gpu_x.data, 
                       gpu_y.data,
                       gpu_z.data,
                       gpu_m.data)
                       
    event.wait()    
    result = gpu_result.get()
    queue.finish()

    return result
