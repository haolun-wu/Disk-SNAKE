import torch
import timeit


size = 4096
a = torch.randn(size, device="cuda:0")


def test_cuda_idx():
    index = torch.randint(0, size, (size,), device="cuda:0")
    b = a[index]
    return b.sum()


def test_cpu_idx():
    index = torch.randint(0, size, (size,), device="cpu")
    b = a[index]
    return b.sum()


def test_cuda_mask():
    mask = torch.randint(0, 2, (size,), device="cuda:0", dtype=torch.bool)
    b = a[mask]
    return b.sum()


def test_cpu_mask():
    mask = torch.randint(0, 2, (size,), device="cpu", dtype=torch.bool)
    b = a[mask]
    return b.sum()


print(f"idx cuda {timeit.timeit(test_cuda_idx, number=1000):.1e}", end="\t")
print(f"cpu {timeit.timeit(test_cpu_idx, number=1000):.1e}", end="\n")
print(f"mask cuda {timeit.timeit(test_cuda_mask, number=1000):.1e}", end="\t")
print(f"cpu {timeit.timeit(test_cpu_mask, number=1000):.1e}", end="\n")

"""
Conclusions:
- Masking is faster than indexing
- GPU is much faster than CPU in 2D
- GPU is slightly faster than CPU in 1D (might be even worse for 4096 tensor)
"""