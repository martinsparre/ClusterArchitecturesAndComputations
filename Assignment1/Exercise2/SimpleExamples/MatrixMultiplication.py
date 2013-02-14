import time
import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
   
def get_kernel(ctx, xdim):
    """
    This returns a compiled kernels
    """
    kernel_src_template = """
    #define RESULT(y,x) result[(y*XDIM)+x]
    #define XDIM %(XDIM)i
    #define MATRIX(y,x) matrix[(y*XDIM)+x]
    #define MATRIX2(y,x) matrix2[(y*XDIM)+x]
   
    __kernel void add(__global float *result,
                      __global float *matrix,
                      __global float *matrix2) {
                       
       int x = get_global_id(0);
       int y = get_global_id(1);
   
       RESULT(y,x) = MATRIX(y,x) + MATRIX2(y,x);
    }
    """
                                 
    kernel_src = kernel_src_template % {
                     'XDIM' : xdim
                 }
                 
    kernel = cl.Program(ctx, kernel_src).build()
                                                                                           
    return kernel

def main():
    # Allocate the first GPU
    ctx = cl.create_some_context(0)#use device 0, the GPU
    queue = cl.CommandQueue(ctx)
    
    # Define dimensions
    ydim = 1024
    xdim = 1024

    # Create random matrix
    matrix = np.random.random((ydim, xdim))
    matrix = np.float32(matrix)

    # Create random matrix2
    matrix2 = np.random.random((ydim, xdim))
    matrix2 = np.float32(matrix2)

    # Get the compiled kernel
    kernel = get_kernel(ctx, xdim)

    # Start timing
    t1 = time.time()
    
    # Move data to the GPU
    gpu_matrix = cl_array.to_device(queue, matrix)
    gpu_matrix2 = cl_array.to_device(queue, matrix2)
    gpu_result = cl_array.zeros(queue, (ydim, xdim), np.float32)

    # Define grid shape (the same as the matrix dimensions)
    grid_shape = (ydim, xdim)
    
    # Get group shape based on the matrix dimensions and the actual hardware
    group_shape = (16,16)#(32,16)
    
    # Execute the kernel
    event = kernel.add(queue, 
                       grid_shape, group_shape, 
                       gpu_result.data, 
                       gpu_matrix.data, 
                       gpu_matrix2.data)
                       
    # Wait for the kernel to finish
    event.wait()
    
    # Move the result from GPU to CPU
    result = gpu_result.get()
    
    # Measure end time
    t2 = time.time()

    # Print result and execution time
    print result
    print "Elapsed: %f seconds " % (t2-t1)

    # Free the GPU resource
    queue.finish()



if __name__ == "__main__":
    main()
