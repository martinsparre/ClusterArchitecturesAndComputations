import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
import time

    
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

def main():
    xdim = ydim = 8


    
    # Create CPU matrix data
    vector = np.float32(xrange(xdim))+1
    print vector
    
    # Create PyOpenCL context
    ctx = cl.create_some_context(0)
    queue = cl.CommandQueue(ctx)

    # Start timer
    t1 = time.time()
    
    # Move CPU matrix to GPU
    
    gpu_vector = cl_array.to_device(queue, vector)
    gpu_result = cl_array.zeros(queue, (ydim, xdim), np.float32)    
    

    
    # Get GPU kernel
    kernel = get_kernel_OuterProduct(queue)
    
    # Invoke OpenCL kernel (queue, global_shape, local_shape, data)
    event = kernel.outer(queue, 
                            gpu_result.shape,
                            None,
                            gpu_vector.data,
                            gpu_result.data)

    # Wait for kernel to finish
    event.wait()
    
    # Retrieve result from gpu 
    result = gpu_result.get()

    # End timer
    t2 = time.time()
    
    print ""
 #   print "input: \n%s" % (matrix)
    print ""
    print "result: \n%s" % (result)
    print ""
    print "Execution time including transfers: %s seconds" % (t2-t1)

    # Free GPU resource
    queue.finish()
    
if __name__ == "__main__":
        main()

