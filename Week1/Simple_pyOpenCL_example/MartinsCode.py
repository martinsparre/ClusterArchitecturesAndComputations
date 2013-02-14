#Port from Adventures in OpenCL Part1 to PyOpenCL
# http://enja.org/2010/07/13/adventures-in-opencl-part-1-getting-started/
# http://documen.tician.de/pyopencl/

import pyopencl as cl
import numpy

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

fstr = """
__kernel void part1(__global float* a, __global float* b, __global float* c)
{
    unsigned int i = get_global_id(0);

    c[i] = a[i] + b[i];
}
"""

program = cl.Program(ctx, fstr).build()

mf = cl.mem_flags

#initialize client side (CPU) arrays
a = numpy.array(range(10), dtype=numpy.float32)
b = numpy.array(range(10), dtype=numpy.float32)

#create OpenCL buffers
a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
dest_buf = cl.Buffer(ctx, mf.WRITE_ONLY, b.nbytes)

program.part1(queue, a.shape, None, a_buf, b_buf, dest_buf)
c = numpy.empty_like(a)
#cl.enqueue_read_buffer(queue, dest_buf, c).wait()
cl.enqueue_copy(queue, c,dest_buf)
print "a", a
print "b", b
print "c", c


