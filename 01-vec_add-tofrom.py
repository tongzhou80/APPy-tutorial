import numpy as np
import numba
import torch
import appy
from appy.utils import allclose, bench

# APPy works with both `torch` tensors and `cupy` ndarrays.
# Use @appy.jit(lib='cupy') work with `cupy` ndarrays.

@appy.jit
def kernel_appy(a, b):
    c = np.empty_like(a)
    #pragma parallel for simd to(a,b,c) from(c)
    for i in range(a.shape[0]):
        c[i] = a[i] + b[i]
    return c

@numba.njit(parallel=True)
def kernel_numba(a, b):
    c = np.empty_like(a)
    for i in numba.prange(a.shape[0]):
        c[i] = a[i] + b[i]
    return c

def test():
    for N in [10000, 100000, 1000000, 10000000, 100000000]:
        a = np.random.randn(N)
        b = np.random.randn(N)
        c_numba = kernel_numba(a, b)
        print(f"N: {a.shape[0]}, dtype: {a.dtype}")

        c_appy = kernel_appy(a, b)
        assert allclose(c_appy, c_numba, atol=1e-6)
        numba_time = bench(lambda: kernel_numba(a, b))
        appy_time = bench(lambda: kernel_appy(a, b))
        print(f"kernel_numba: {numba_time:.4f} ms")
        print(f"kernel_appy: {appy_time:.4f} ms")
        print(f'speedup over numba: {(numba_time/appy_time):.2f}\n')


if __name__ == "__main__":
    test()
