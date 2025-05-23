# Programming Your GPU with APPy

This is a hands-on tutorial that introduces the basics of running your Python code on the GPUs 
through a series of worked examples.

Inspired by the [openmp-tutorial](https://github.com/UoB-HPC/openmp-tutorial), we include the following examples:

* `vadd` – A simple vector addition program, often considered the "hello world" of GPU programming.
* `pi` – A numerical integration program that calculates and approximate value of π.
* `jac_solv` – A Jacobi solver.
* `heat` - An explicit finite difference 5-point stencil code.
* `spmv` - An example of parallelizing the outer loop and vectorizing the inner loop.

# Packages Required

APPy can be installed using `pip` (APPy relies tools in `torch` so `torch` needs to be installed too). `numpy` and `numba` also need to installed for input setup and performance comparisons.

```bash
pip install numpy numba torch appyc
```