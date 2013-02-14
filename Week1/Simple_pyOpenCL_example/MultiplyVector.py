import time
import numpy
import pyopencl as cl

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

fstr = """
__kernel void part1(__global const float* a, __global float* c)
{
    unsigned int i = get_global_id(0);

    c[i] = 2.0*a[i];
}
"""

program = cl.Program(ctx, fstr).build()

Data = numpy.random.random(1000000)
Data = numpy.array(Data,dtype=numpy.float32)

mf = cl.mem_flags
DataBuffer = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=Data)
ResBuffer =  cl.Buffer(ctx, mf.WRITE_ONLY, Data.nbytes)

Start = time.time()
program.part1(queue, Data.shape, None, DataBuffer, ResBuffer)

Res = numpy.empty_like(Data)
cl.enqueue_copy(queue,Res,ResBuffer)

End = time.time()

print 'Time',End-Start

print Res
print Data[0],Res[0]
