import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array

def OuterProduct(ctx,queue,vector):
    Length = len(vector)

    gpu_vector = cl_array.to_device(queue, vector)
    gpu_result = cl_array.zeros(queue, (Length,Length), np.float32)    
    kernel = get_kernel_OuterProduct(queue)
    event = kernel.outer(queue, 
                            gpu_result.shape,
                            (16,16),
                            gpu_vector.data,
                            gpu_result.data)

    event.wait()    
    result = gpu_result.get()
    queue.finish()
    return result

def get_kernel_OuterProduct(queue):
    """
    This function generates the kernel source from the template with constants replaces by actual values
    """

    # Replace kernel source template constants with actual constants
    kernel_src = """
           #define OUTER(y,x) OuterProduct[(y*get_global_size(0))+x]

           __kernel void outer(__global float *vector,__global float *OuterProduct) {
               int x = get_global_id(0);
               int y = get_global_id(1);

               OUTER(y,x) = vector[x] * vector[y];
           }
    """
    
    # Check if we are using NVIDIA GPUs
    if "NVIDIA" in queue.device.vendor:
        options = "-cl-mad-enable -cl-fast-relaxed-math"
    else:
        options = ""


    print "------------------------- Kernel src --------------------------"
    print kernel_src
    print "---------------------------------------------------------------"
    print "compile options: %s" % (options)
    print "---------------------------------------------------------------"
                                     
    kernel = cl.Program(queue.context, kernel_src).build(options=options)
    
    return kernel



   
def get_kernel(ctx, xdim,GRAV_CONST=1.0):
    """
    This returns a compiled kernels
    """
    kernel_src_template = """

    #define GRAV_CONST %(GRAV_CONST)f
    #define XDIM %(XDIM)i
    #define RESULT(y,x) result[(y*XDIM)+x]
    #define MASS2(y,x) mass2[(y*XDIM)+x]
    #define RAD2(y,x) rad2[(y*XDIM)+x]
   
    __kernel void CalcF(__global float *result,
                      __global float *mass2,
                      __global float *rad2) {
                       
       int x = get_global_id(0);
       int y = get_global_id(1);
   
       RESULT(y,x) = GRAV_CONST*MASS2(y,x)  / pow(RAD2(y,x),(float)1.5) ;//
    }
    """
                                 #
    kernel_src = kernel_src_template % {
                     'XDIM' : xdim,
                     'GRAV_CONST' : GRAV_CONST
                 }
    print kernel_src
                 
    kernel = cl.Program(ctx, kernel_src).build()
                                                                                           
    return kernel

def CalcF(ctx,queue,m2,r2):

 
    # Define dimensions
    xdim = ydim = m2.shape[0]


#    m2 = np.float32(m2)
#    r2 = np.float32(r2)
 
    # Get the compiled kernel
    kernel = get_kernel(ctx, xdim)
    
    # Move data to the GPU

    gpu_m2 = cl_array.to_device(queue, m2)
    gpu_r2 = cl_array.to_device(queue, r2)
    gpu_result = cl_array.zeros(queue, (ydim, xdim), np.float32)

    # Define grid shape (the same as the matrix dimensions)
    grid_shape = (ydim, xdim)
    
    # Get group shape based on the matrix dimensions and the actual hardware
    group_shape = (32,16)
    

    event = kernel.CalcF(queue, 
                       grid_shape, group_shape, 
                       gpu_result.data, 
                       gpu_m2.data, 
                       gpu_r2.data)
                       
    event.wait()    
    result = gpu_result.get()
    queue.finish()

    return result
