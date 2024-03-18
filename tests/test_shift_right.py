import torch
import sys
import os

parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent)
import timeit
from kbgen.utils import shift_right, get_gpu_status

torch.set_default_device("cuda")

get_gpu_status()
print("Generating data...")
batch_size = 4096
max_len = 10
x = torch.randint(0, 50000, (batch_size, max_len), dtype=torch.int)
print(f"x {x.shape} memory usage", x.nelement() * x.element_size() / 1e6, "MB")
get_gpu_status()

print("Shifting right...")
y = shift_right(x)
print(f"x {x.shape} memory usage", x.nelement() * x.element_size() / 1e6, "MB")
get_gpu_status()

print("Timing...")
print(
    "shift_right inplace=True",
    timeit.timeit(
        "shift_right(x, inplace=True)",
        setup="x=torch.randint(0, 50000, (4096, 10), dtype=torch.int)",
        number=100,
        globals=globals(),
    ),
)
print(
    "shift_right inplace=False",
    timeit.timeit(
        "shift_right(x, inplace=False)",
        setup="x=torch.randint(0, 50000, (4096, 10), dtype=torch.int)",
        number=100,
        globals=globals(),
    ),
)

print(
    "shift_right inplace=True",
    timeit.timeit(
        "shift_right(x, inplace=True)",
        setup="x=torch.randint(0, 50000, (4096, 10), dtype=torch.int)",
        number=100,
        globals=globals(),
    ),
)

print(
    "shift_right inplace=False",
    timeit.timeit(
        "shift_right(x, inplace=False)",
        setup="x=torch.randint(0, 50000, (4096, 10), dtype=torch.int)",
        number=100,
        globals=globals(),
    ),
)
