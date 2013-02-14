import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
import time

    
def get_kernel(queue):
    """
    This function generates the kernel source from the template with constants replaces by actual values
    """
    kernel_source_template = get_kernel_source_template()

    # Replace kernel source template constants with actual constants
    kernel_src = """
           #define RESULT(y,x) result[(y*get_global_size(0))+x]

           __kernel void multiply(__global float *result) {
               int x = get_global_id(0);
               int y = get_global_id(1);

               RESULT(y,x) *= 2;
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
    ydim = 2048
    xdim = 2048
    mul_factor = 2
    
    # Create CPU matrix data
    matrix = np.float32(np.random.random((ydim, xdim)))
    
    # Create PyOpenCL context
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    # Start timer
    t1 = time.time()
    
    # Move CPU matrix to GPU
    gpu_matrix = cl_array.to_device(queue, matrix)
    
    # Get GPU kernel
    kernel = get_kernel(queue)
    
    # Invoke OpenCL kernel (queue, global_shape, local_shape, data)
    event = kernel.multiply(queue, 
                            gpu_matrix.shape,
                            (16, 16),
                            gpu_matrix.data)

    # Wait for kernel to finish
    event.wait()
    
    # Retrieve result from gpu 
    result = gpu_matrix.get()

    # End timer
    t2 = time.time()
    
    print ""
    print "input: \n%s" % (matrix)
    print ""
    print "result: \n%s" % (result)
    print ""
    print "Execution time including transfers: %s seconds" % (t2-t1)

    # Free GPU resource
    queue.finish()
    
if __name__ == "__main__":
        main()

