import numpy as np
from numba import jit, prange
import time
import torch
import appy
from appy.utils import allclose, bench

# Constants
TOLERANCE = 0.001
DEF_SIZE = 1000
MAX_ITERS = 100000
LARGE = 1000000.0


def init_diag_dom_near_identity_matrix(Ndim):
    """Generate a random, diagonally dominant matrix near identity."""
    A = np.random.randint(0, 23, size=(Ndim, Ndim)) / 1000.0
    for i in range(Ndim):
        row_sum = np.sum(A[i, :])
        A[i, i] += row_sum  # Make diagonal dominant
        A[i, :] /= row_sum  # Scale the row to be near identity
    return A


@jit(nopython=True, parallel=True)
def jacobi_solver_numba(A, b, x1, x2, Ndim):
    """Perform Jacobi iterative method."""
    iters = 0
    conv = LARGE

    while (conv > TOLERANCE * TOLERANCE) and (iters < MAX_ITERS):
        # Update x2 based on Jacobi iteration
        for i in prange(Ndim):
            sum_val = 0.0
            for j in range(Ndim):
                sum_val += A[i, j] * x1[j] * (i != j)
            x2[i] = (b[i] - sum_val) / A[i, i]

        # Calculate convergence
        conv = 0.0
        for i in prange(Ndim):
            tmp = x2[i] - x1[i]
            conv += tmp * tmp

        # Swap x1 and x2
        x1, x2 = x2, x1
        iters += 1

    return x1, conv, iters


@appy.jit
def jacobi_solver_appy(A, b, x1, x2, Ndim):
    """Perform Jacobi iterative method."""
    TOLERANCE = 0.001
    DEF_SIZE = 1000
    MAX_ITERS = 100000
    LARGE = 1000000.0
    iters = 0
    conv = torch.tensor([LARGE], device='cuda')

    while (conv[0] > TOLERANCE * TOLERANCE) and (iters < MAX_ITERS):
        # Update x2 based on Jacobi iteration
        #pragma parallel for
        for i in range(Ndim):
            x2[i] = 0.0
            #pragma simd
            for j in range(Ndim):
                x2[i] += A[i, j] * x1[j] * (i != j)
            x2[i] = (b[i] - x2[i]) / A[i, i]

        # Calculate convergence
        conv[0] = 0.0
        #pragma parallel for simd
        for i in range(Ndim):
            tmp = x2[i] - x1[i]
            #pragma atomic
            conv[0] += tmp * tmp

        # Swap x1 and x2
        x1, x2 = x2, x1
        iters += 1

    return x1, conv[0], iters


def main(Ndim=DEF_SIZE):
    """Main function to run Jacobi solver."""
    print(f"\n\nJacobi solver parallel (Numba + prange version): ndim = {Ndim}")

    # Matrix and vector initialization
    A = init_diag_dom_near_identity_matrix(Ndim)
    b = np.random.rand(Ndim)
    x1 = np.zeros(Ndim)
    x2 = np.zeros(Ndim)

    # Start timer
    start_time = time.time()

    # Run Jacobi solver
    x_final, conv, iters = jacobi_solver_numba(A, b, x1, x2, Ndim)

    # End timer
    elapsed_time = time.time() - start_time
    conv = np.sqrt(conv)

    print(f"Convergence = {conv:.6g} with {iters} iterations and {elapsed_time:.6f} seconds")

    # Verify solution
    x_check = A @ x_final
    err = np.linalg.norm(x_check - b)
    chksum = np.sum(x_final)

    print(f"Jacobi solver: err = {err:.6g}, solution checksum = {chksum:.6g}")
    if err > TOLERANCE:
        print(f"\nWARNING: final solution error > {TOLERANCE}\n\n")

    print(f"\n\nJacobi solver parallel (APPy version): ndim = {Ndim}")

    # Matrix and vector initialization
    x1 = np.zeros(Ndim)
    x2 = np.zeros(Ndim)

    # Start timer
    start_time = time.time()

    # Run Jacobi solver
    A_gpu, b_gpu, x1_gpu, x2_gpu = torch.from_numpy(A).to('cuda'), torch.from_numpy(b).to('cuda'), torch.from_numpy(x1).to('cuda'), torch.from_numpy(x2).to('cuda')
    x_final, conv, iters = jacobi_solver_appy(A_gpu, b_gpu, x1_gpu, x2_gpu, Ndim)
    conv = np.sqrt(conv.item())
    x_final = x_final.cpu().numpy()

    # End timer
    elapsed_time = time.time() - start_time
    
    print(f"Convergence = {conv:.6g} with {iters} iterations and {elapsed_time:.6f} seconds")

    # Verify solution
    x_check = A @ x_final
    err = np.linalg.norm(x_check - b)
    chksum = np.sum(x_final)

    print(f"Jacobi solver: err = {err:.6g}, solution checksum = {chksum:.6g}")
    if err > TOLERANCE:
        print(f"\nWARNING: final solution error > {TOLERANCE}\n\n")


if __name__ == "__main__":
    import sys
    Ndim = int(sys.argv[1]) if len(sys.argv) == 2 else DEF_SIZE
    main(Ndim)
