import numpy as np
import numba
import torch
import appy
from appy.utils import allclose, bench

@numba.njit(parallel=True)
def kernel_numba(n):
    step = 1.0 / n
    s = 0.0
    for i in numba.prange(1, n+1):
        x = (i - 0.5) * step
        s += 4.0 / (1.0 + x * x)
    pi = step * s
    return pi

@appy.jit
def kernel_appy(n):
    step = 1.0 / n
    s = torch.zeros(1, device='cuda')
    #pragma parallel for simd
    for i in range(1, n+1):
        x = (i - 0.5) * step
        #pragma atomic
        s[0] += 4.0 / (1.0 + x * x)
    pi = step * s[0]
    return pi

def test():
    for n in [10000, 100000, 1000000, 10000000]:
        pi_numba = kernel_numba(n)
        pi_appy = kernel_appy(n)
        print(pi_numba, pi_appy)
        assert allclose(pi_appy.cpu().numpy(), np.array([pi_numba]), atol=1e-6)
        numba_time = bench(lambda: kernel_numba(n))
        appy_time = bench(lambda: kernel_appy(n))
        print(f"kernel_numba: {numba_time:.4f} ms")
        print(f"kernel_appy: {appy_time:.4f} ms")
        print(f'speedup over numba: {(numba_time/appy_time):.2f}\n')

if __name__ == "__main__":
    test()
