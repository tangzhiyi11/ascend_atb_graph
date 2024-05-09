import torch
import torch_dipu

from torch.profiler import record_function

import acl

import ctypes

a1 = torch.randn(1, 4096, dtype=torch.float16, device='cuda')
b1 = torch.randn(4096, 4096, dtype=torch.float16, device='cuda')
a2 = torch.randn(1, 4096, dtype=torch.float16, device='cuda')
b2 = torch.randn(4096, 4096, dtype=torch.float16, device='cuda')
out = torch.randn(1, 4096, dtype=torch.float16, device='cuda')
out2 = torch.randn(1, 4096, dtype=torch.float16, device='cuda')

stream, ret = acl.rt.create_stream()
print(stream)

inputs = [a1, b1, a2, b2]
inputs_ptr = [x.data_ptr() for x in inputs]
ctype_inputs = (ctypes.c_void_p * len(inputs))(*inputs_ptr)

outputs = [out]
outputs_ptr = [x.data_ptr() for x in outputs]
ctype_outputs = (ctypes.c_void_p * len(outputs))(*outputs_ptr)

graph = ctypes.CDLL('atb_graph.so')
graph.init(ctypes.c_void_p(None), ctypes.c_void_p(stream))

graph.run(ctype_inputs, len(inputs), ctype_outputs, len(outputs))
print('after compute!!')
print(out)
print()

graph.run(ctype_inputs, len(inputs), ctype_outputs, len(outputs))
print('after compute!!')
print(out)
print()


mm1 = torch.mm(a1.to(torch.float), b1.to(torch.float))
mm2 = torch.mm(a2.to(torch.float), b2.to(torch.float))
add = mm1 + mm2
print('pytorch out:')
print(add)
