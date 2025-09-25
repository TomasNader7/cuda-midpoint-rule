# CUDA Midpoint Rule Numerical Integration

This repository contains a C and CUDA implementation of the midpoint rule for estimating the value of \u03c0 by numerically integrating the function 4/(1+x^2) on the interval [0, 1]. It was completed as part of CSC 630/730 Assignment #3 at the University of Southern Mississippi.

## Files

- `Parallel.cu` – A C/CUDA program that includes both a serial baseline and a parallel implementation using 1‑D CUDA thread blocks.
- `Assign3_CUDA_Trap.pdf` – The assignment description outlining the objectives, requirements, and grading rubric.
- `Report (2).pdf` – A report summarizing the design of the algorithm, key implementation details, performance results, and conclusions.

## Building

To compile the CUDA program, ensure you have the NVIDIA CUDA Toolkit installed. Then run:

```bash
nvcc -O2 -arch=sm_30 -o Parallel Parallel.cu
```

On a system with compute capability ≥ 6.0 (supports double precision atomics), you can compile with a higher architecture flag:

```bash
nvcc -O3 -arch=sm_60 -o Parallel Parallel.cu
```

## Running

The program takes two command‑line arguments:

```
./Parallel <n> <threads_per_block>
```

- `n` – Number of subintervals used in the integration.
- `threads_per_block` – Threads per CUDA block (1‑D block).

Example:

```bash
./Parallel 1000000 256
```

The program prints both the serial and parallel estimates of \u03c0, the runtimes of each implementation, and the resulting speedup.

## Results

Refer to the report for a full discussion of results. In summary, the parallel version yields substantial speedups compared to the serial baseline once the problem size is large enough to amortize kernel launch overhead, while still providing an accurate estimate of \u03c0.

## License

This project is provided for educational purposes.
