import torch

import sys

a = torch.randn(128, 128)
b = torch.randn(128, 128)

for i in range(10):
    on_cpu = torch.matmul(a, b)
    on_gpu = torch.matmul(a.cuda(), b.cuda())
    if not on_cpu.isclose(on_gpu.cpu(), atol=1e-3, rtol=1e-3).all():
        sys.exit(1)
    print("Success")