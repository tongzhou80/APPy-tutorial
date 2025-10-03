import numpy as np
import time
from numba import njit, prange
from appy import jit, to_gpu, to_cpu
import torch

N = 20_000_000
TOL = 1e-7

#
#  This is a simple program to add two vectors
#  and verify the results.
#
#  History: Written by Tim Mattson, November 2017
#  Python version created by ChatGPT based on the original version
#

@njit(parallel=True)
def initialize_arrays(a, b, c, res):
    for i in prange(N):
        a[i] = float(i)
        b[i] = 2.0 * float(i)
        c[i] = 0.0
        res[i] = a[i] + b[i]

# @jit(entry_to_device='a,b')
# def add_vectors(a, b, c):
#     c_cpu = torch.from_numpy(c)
#     c = to_gpu(c)
#     for i in prange(N):
#         c[i] = a[i] + b[i]
#     c_cpu.copy_(c)

@jit
def add_vectors(a, b, c):
    for i in prange(N):
        c[i] = a[i] + b[i]

@njit(parallel=True)
def test_result(c, res):
    err = 0
    for i in prange(N):
        val = c[i] - res[i]
        val = val * val
        if val > TOL:
            err += 1
            print(res[i], c[i])
    return err

def main():
    a = np.empty(N, dtype=np.float32)
    b = np.empty(N, dtype=np.float32)
    c = np.empty(N, dtype=np.float32)
    res = np.empty(N, dtype=np.float32)

    start_time = time.time()
    init_start = time.time()
    initialize_arrays(a, b, c, res)
    init_time = time.time() - init_start

    compute_start = time.time()
    add_vectors(a, b, c)
    compute_time = time.time() - compute_start

    test_start = time.time()
    err = test_result(c, res)
    test_time = time.time() - test_start

    total_time = time.time() - start_time

    print(f"vectors added with {err} errors")
    print(f"Init time:    {init_time:.3f}s")
    print(f"Compute time: {compute_time:.3f}s")
    print(f"Test time:    {test_time:.3f}s")
    print(f"Total time:   {total_time:.3f}s")

if __name__ == "__main__":
    main()
    main()
    main()