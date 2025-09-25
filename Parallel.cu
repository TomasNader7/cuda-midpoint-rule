/*
* File: Parallel.cu
* Course: CSC 630/730 - Assignment 3
* Purpose: Compute π by numerical integration of ∫_0^1 4/(1+x^2) dx
*          using the adpoint rule, with BOTH:
*              - (1) Seria C implementation (baseline Ts)
*              - (2) CUDA arallel implementation (Tp) with 1-D blocks.
*
* Compile: 
*		- module load cuda-toolkit
*		- nvcc -O2 -arch=sm_30 -o Parallel Parallel.cu
*		- srun -p gpu --gres gpu:1 -n 1 -N 1 --pty --mem 1000 -t 2:00 bash
*		- ./Parallel 100 128
*		- exit
*
* Run examples (Magnolia shell):
*	./Parallel <n> <threads_per_bk>
*	./Parallel 100 128
*	./Parallel 1000 256
*	./Parallel 10000 256
*
* Input:
*		<n>               Positive integer number of subintervals.
*		<threads_per_bk>  Positive integer threads per block (1-D).
*
* Outputs (printed, 6 decimals where applicable):
*   n, block_ct, threads_per_bk, threads_total
*   Serial integral
*   Parallel integral
*   Ts (serial seconds), Tp (CUDA seconds), Speedup S = Ts/Tp
*
* Expected output format (example fields; numbers will vary):
*   n=1000, block_ct=4, threads_per_bk=256, threads_total=1024
*   Serial integral   : 3.141593
*   Parallel integral : 3.141592
*   Ts (serial, s)    : 0.000090
*   Tp (CUDA, s)      : 0.000700
*   Speedup S=Ts/Tp   : 0.129000
*
* Notes:
*		(1) GPU accumulator uses float* for broad device compatibility; final result
*		is converted to double on host. Expect tiny differences (~1e-6) vs serial.
*		(2) If target GPU supports double atomics (SM ≥ 6.0), the accumulator can be
*		double* instead of float*.
*		(3) atomicAdd(double*,double) compile error: use -arch=sm_60+ OR keep float* accumulator.
*		(4) Serial Ts measured with clock() around midpoint_integral(n).
*		(5) Parallel Tp measured with CUDA events around the kernel.
*		(6) If the GPU supports SM ≥ 60 (double atomics):
*		nvcc -O3 -std=c++11 -arch=sm_60 Parallel.cu -o Parallel
* 
* Author: Tomas Nader
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

// Helper function. Can be called from __host__ or __device__ 
__host__ __device__ inline double f(double x) {
	return 4.0 / (1.0 + x * x);
}

/* ----------------------------------------------------------------------------------------------
*											Host Code
* -----------------------------------------------------------------------------------------------
*/
double midpoint_integral(int n) {
	const double a = 0.0, b = 1.0;
	const double h = (b - a) / (double)n;
	double sum = 0.0;

	for (int i = 0; i < n; ++i) {
		double x = (i + 0.5) * h;       // midpoint of subinterval i
		sum += f(x);                       // add to running total
	}
	return h * sum;
}
/* ----------------------------------------------------------------------------------------------
*											Kernel and Device Code
* -----------------------------------------------------------------------------------------------
*/

__global__ void midpoint_kernel(int n, double h, float* sum_accum) {
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	int threads_total = gridDim.x * blockDim.x;
	double local_sum = 0.0;

	for (int i = gid; i < n; i += threads_total) {
		double x = (i + 0.5) * h;       // midpoint of subinterval 
		local_sum += f(x);              
	}
	atomicAdd(sum_accum, (float)local_sum);

}

/* ----------------------------------------------------------------------------------------------
*											main
* -----------------------------------------------------------------------------------------------
*/
int main(int argc, char* argv[]) {
	if (argc != 3) {
		fprintf(stderr, "Usage: %s <n> <threads_per_bk>\n", argv[0]);
		return 1;
	}

	// Derive integration and launch parameters
	int n = atoi(argv[1]);
	int thd_per_blk = atoi(argv[2]);
	if (n <= 0 || thd_per_blk <= 0) { fprintf(stderr, "n and threads_per_bk must be > 0\n"); return 1; }

	double h = 1.0 / (double)n;
	int blk_ct = (n + thd_per_blk - 1) / thd_per_blk;
	int threads_total = blk_ct * thd_per_blk;

	// Measure the serial baseline (Ts)
	clock_t t0 = clock();
	double serial_integral = midpoint_integral(n);
	clock_t t1 = clock();
	double Ts = (double)(t1 - t0) / CLOCKS_PER_SEC;

	// Allocate GPU memory for the shared sum (no-h) and zero it
	float* sum_no_h = NULL;
	cudaError_t err = cudaMallocManaged(&sum_no_h, sizeof(double));
	if (err != cudaSuccess) { fprintf(stderr, "cudaMallocManaged error = %s\n",
		cudaGetErrorString(err));
	fprintf(stderr, "Quitting\n");
	exit(-1); 
	}

	*sum_no_h = 0.0f;

	// Time the kernel (Tp) with CUDA events
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	midpoint_kernel <<< blk_ct, thd_per_blk >>> (n, h, sum_no_h);
	cudaError_t kerr = cudaGetLastError();
	if (kerr != cudaSuccess) {
		fprintf(stderr, "Kernel launch error = %s\n", cudaGetErrorString(kerr));
		return 1;
	}

	cudaDeviceSynchronize();
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float ms = 0.0f;
	cudaEventElapsedTime(&ms, start, stop);
	double Tp = ms / 1000.0;

	// Scale and compute speedup

	double parallel_integral = h * (double)(*sum_no_h);
	double speedup = Ts / Tp;

	// Print results
	printf("n=%d, block_ct=%d, threads_per_bk=%d, threads_total=%d\n",
		n, blk_ct, thd_per_blk, threads_total);
	printf("Serial integral   : %.6f\n", serial_integral);
	printf("Parallel integral : %.6f\n", parallel_integral);
	printf("Ts (serial, s)    : %.6f\n", Ts);
	printf("Tp (CUDA, s)      : %.6f\n", Tp);
	printf("Speedup S=Ts/Tp   : %.6f\n", speedup);


	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFree(sum_no_h);
	return 0;
	
}

