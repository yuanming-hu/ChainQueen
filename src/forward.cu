#include "kernels.h"
#include "linalg.cuh"
#include <cstdio>
#include "../../../../../../../usr/local/cuda/include/driver_types.h"

void run(real *a, real *b, real *c) {

}

__global__ void saxpy_g(int n, real a, real *x, real *y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] = a * x[i] + y[i];
  }
}

void saxpy_cuda(int N, real alpha, real *x, real *y) {
  real* d_x, *d_y;
  /*
  cudaMalloc(&d_x, n * sizeof(real));
  cudaMalloc(&d_y, n * sizeof(real));
  cudaMemcpy(d_x, x, n * sizeof(real), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, n * sizeof(real), cudaMemcpyHostToDevice);
  for (int i = 0; i < n; i++) {
    printf("%f %f\n", x[i], y[i]);
  }
  printf("size %d\n", n * sizeof(real));
  saxpy_g<<<1, 256>>>(n, alpha, x, y);
  cudaMemcpy(y, d_y, n * sizeof(real), cudaMemcpyDeviceToHost);
  for (int i = 0; i < n; i++) {
    printf("%f %f\n", x[i], y[i]);
  }
  cudaFree(d_x);
  cudaFree(d_y);
  printf("done\n");
  */

  cudaMalloc(&d_x, N*sizeof(float));
  cudaMalloc(&d_y, N*sizeof(float));

  cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);

  saxpy_g<<<(N+255)/256, 256>>>(N, 2.0f, d_x, d_y);

  cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);

  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = max(maxError, abs(y[i]-i * 4));
  printf("Max error: %f\n", maxError);

  cudaFree(d_x);
  cudaFree(d_y);
}

void test() {
  int N = 256;
  float *x, *y, *d_x, *d_y;
  x = (float*)malloc(N*sizeof(float));
  y = (float*)malloc(N*sizeof(float));

  cudaMalloc(&d_x, N*sizeof(float));
  cudaMalloc(&d_y, N*sizeof(float));

  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);

  saxpy_g<<<(N+255)/256, 256>>>(N, 2.0f, d_x, d_y);

  cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);

  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = max(maxError, abs(y[i]-4.0f));
  printf("Max error: %f\n", maxError);

  cudaFree(d_x);
  cudaFree(d_y);
  free(x);
  free(y);
}
