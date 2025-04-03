import numpy as np
import numba
from numba import jit, prange
import time

PI = np.arccos(-1.0)  # Pi

def initial_value(n, dx, length):
    u = np.zeros((n, n), dtype=np.float64)
    y = dx
    for j in range(n):
        x = dx
        for i in range(n):
            u[i, j] = np.sin(PI * x / length) * np.sin(PI * y / length)
            x += dx
        y += dx
    return u

@jit(nopython=True, parallel=True)
def solve(n, alpha, dx, dt, u, u_tmp):
    r = alpha * dt / (dx * dx)
    r2 = 1.0 - 4.0 * r
    
    for i in prange(n):
        for j in range(n):
            u_tmp[i, j] = (
                r2 * u[i, j] +
                r * (u[i+1, j] if i < n-1 else 0.0) +
                r * (u[i-1, j] if i > 0 else 0.0) +
                r * (u[i, j+1] if j < n-1 else 0.0) +
                r * (u[i, j-1] if j > 0 else 0.0)
            )

@jit(nopython=True)
def solution(t, x, y, alpha, length):
    return np.exp(-2.0 * alpha * PI * PI * t / (length * length)) * np.sin(PI * x / length) * np.sin(PI * y / length)

@jit(nopython=True)
def l2norm(n, u, nsteps, dt, alpha, dx, length):
    time = dt * nsteps
    l2norm_val = 0.0
    y = dx
    for j in range(n):
        x = dx
        for i in range(n):
            answer = solution(time, x, y, alpha, length)
            l2norm_val += (u[i, j] - answer) ** 2
            x += dx
        y += dx
    return np.sqrt(l2norm_val)

if __name__ == "__main__":
    import sys
    
    n = 1000
    nsteps = 10
    if len(sys.argv) == 3:
        n = int(sys.argv[1])
        nsteps = int(sys.argv[2])
    
    alpha = 0.1
    length = 1000.0
    dx = length / (n + 1)
    dt = 0.5 / nsteps
    r = alpha * dt / (dx * dx)
    
    print(f"Grid size: {n} x {n}")
    print(f"Alpha: {alpha}")
    print(f"Time step: {dt}")
    print(f"r value: {r} ({'Warning: unstable' if r > 0.5 else 'Stable'})")
    
    u = initial_value(n, dx, length)
    u_tmp = np.zeros_like(u)
    
    start_time = time.time()
    for _ in range(nsteps):
        solve(n, alpha, dx, dt, u, u_tmp)
        u, u_tmp = u_tmp, u  # Swap references
    solve_time = time.time() - start_time
    
    norm = l2norm(n, u, nsteps, dt, alpha, dx, length)
    total_time = time.time() - start_time
    
    print(f"Error (L2 norm): {norm}")
    print(f"Solve time (s): {solve_time}")
    print(f"Total time (s): {total_time}")
