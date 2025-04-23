import numpy as np
from numba import njit, prange
import time

TOLERANCE = 0.001
DEF_SIZE = 1500
MAX_ITERS = 100000
LARGE = 1e6

@njit(parallel=True)
def init_diag_dom_near_identity_matrix(Ndim, rand_vals):
    A = np.zeros((Ndim, Ndim), dtype=np.float64)
    idx = 0
    for i in prange(Ndim):
        sum_row = 0.0
        for j in range(Ndim):
            A[i, j] = rand_vals[idx] / 1000.0
            sum_row += A[i, j]
            idx += 1
        A[i, i] += sum_row
        for j in range(Ndim):
            A[i, j] /= sum_row
    return A

@njit(parallel=True)
def jacobi_iteration(A, b, x1, x2, Ndim):
    conv = LARGE
    iters = 0

    while conv > TOLERANCE * TOLERANCE and iters < MAX_ITERS:
        for i in prange(Ndim):
            temp = 0.0
            for j in range(Ndim):
                temp += (A[i, j] * x1[j]) if i != j else 0.0
            x2[i] = (b[i] - temp) / A[i, i]

        conv = 0.0
        for i in prange(Ndim):
            tmp = x2[i] - x1[i]
            conv += tmp * tmp

        x1, x2 = x2, x1
        iters += 1

    return x1, conv, iters

@njit
def validate_solution(A, x, b):
    Ndim = A.shape[0]
    err = 0.0
    chksum = 0.0
    for i in range(Ndim):
        computed = 0.0
        for j in range(Ndim):
            computed += A[i, j] * x[j]
        diff = computed - b[i]
        err += diff * diff
        chksum += x[i]
    err = np.sqrt(err)
    return err, chksum

def main(Ndim=DEF_SIZE):
    print(f"\n\njacobi solver parallel (numba + parallel): ndim = {Ndim}")

    np.random.seed(0)  # Set the fixed seed here
    rand_vals = np.random.randint(0, 23, size=Ndim * Ndim)
    b = np.random.rand(Ndim) * 0.51

    A = init_diag_dom_near_identity_matrix(Ndim, rand_vals)
    x1 = np.zeros(Ndim, dtype=np.float64)
    x2 = np.zeros(Ndim, dtype=np.float64)

    start_time = time.time()
    x_final, conv, iters = jacobi_iteration(A, b, x1, x2, Ndim)
    elapsed_time = time.time() - start_time

    print(f"Convergence = {np.sqrt(conv)} with {iters} iterations and {elapsed_time:.6f} seconds")

    err, chksum = validate_solution(A, x_final, b)
    print(f"jacobi solver: err = {err:.6f}, solution checksum = {chksum:.6f}")

    if err > TOLERANCE:
        print(f"\nWARNING: final solution error > {TOLERANCE}\n")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        try:
            Ndim = int(sys.argv[1])
        except ValueError:
            print(f"Invalid argument: {sys.argv[1]} is not an integer.")
            sys.exit(1)
    else:
        Ndim = DEF_SIZE
    main(Ndim)
    main(Ndim)