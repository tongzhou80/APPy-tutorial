import numpy as np
import time
import appy

@appy.jit(dump_final_appy=True)
def increment(a):
    for i in appy.prange(a.shape[0]):
        a[i] = a[i] + 1

def main():
    a = np.zeros(10_000_000, dtype=np.int32)
    start_time = time.time()
    increment(a)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")

    assert np.all(a == 1)

if __name__ == "__main__":
    main()
    main()
    main()