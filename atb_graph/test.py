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

inputs = [a1, b1, a2, b2]
inputs_ptr = [x.data_ptr() for x in inputs]
ctype_inputs = (ctypes.c_void_p * len(inputs))(*inputs_ptr)

outputs = [out]
outputs_ptr = [x.data_ptr() for x in outputs]
ctype_outputs = (ctypes.c_void_p * len(outputs))(*outputs_ptr)

graph = ctypes.CDLL('atb_graph.so')


print('before compute!!')
print(a1)
print(a2)
print(out)
print('a1 ptr:', hex(a1.data_ptr()))
print('b1 ptr:', hex(b1.data_ptr()))
print('a2 ptr:', hex(a2.data_ptr()))
print('b2 ptr:', hex(b2.data_ptr()))
print('out ptr:', hex(out.data_ptr()))
print()


stream, ret = acl.rt.create_stream()
print(stream)

graph.init(ctypes.c_void_p(None), ctypes.c_void_p(stream))

path = "/tzy/atb/atb_graph/profiler_atb"

# with torch_dipu.profiler.NativeProfile(path, with_stack=False):
if True:
    with record_function('atb_compute'):
        graph.run(ctype_inputs, len(inputs), ctype_outputs, len(outputs))
        ret = acl.rt.synchronize_stream(stream)
print('after compute!!')
print(out)
print()


outputs2 = [out2]
outputs2_ptr = [x.data_ptr() for x in outputs2]
ctype_outputs2 = (ctypes.c_void_p * len(outputs2))(*outputs2_ptr)

graph.run(ctype_inputs, len(inputs), ctype_outputs2, len(outputs))
graph.run(ctype_inputs, len(inputs), ctype_outputs2, len(outputs))

# ret = acl.rt.synchronize_stream(stream)
print('after compute!!')
print(out2)
print()


mm1 = torch.mm(a1.to(torch.float), b1.to(torch.float))
mm2 = torch.mm(a2.to(torch.float), b2.to(torch.float))
add = mm1 + mm2
print('pytorch out:')
print(add)

print()
print()
print('#########################')

path = "/tzy/atb/atb_graph/profiler_ge"
class OpModule(torch.nn.Module):
    def forward(self, a1, b1, a2, b2):
        mm1 = torch.mm(a1, b1)
        mm2 = torch.mm(a2, b2)
        return mm1 + mm2

model = OpModule()
compiled_model = torch.compile(model, backend='ascendgraph')
out = compiled_model(a1, b1, a2, b2)

out = compiled_model(a1, b1, a2, b2)
print(out)

path = "/tzy/atb/atb_graph/profiler_ge"
print()
print()
print('#########################')


with torch_dipu.profiler.NativeProfile(path, with_stack=False):
    with record_function('ge_compute'):
        out = compiled_model(a1, b1, a2, b2)
