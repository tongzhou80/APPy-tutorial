import numpy as np
import time

def increment(a):
    for i in range(a.shape[0]):
        a[i] += 1

def main():
    a = np.empty(1_000_000, dtype=np.int32)
    start_time = time.time()
    increment(a)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")

    assert np.all(a == 1)

if __name__ == "__main__":
    main()