#include "kernels.h"
#include "linalg.cuh"
#include <cstdio>
#include "../../../../../../../usr/local/cuda/include/driver_types.h"

void run(real *a, real *b, real *c) {

}

__global__ void saxpy_g(int n, real a, real *x, real *y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  x[i] = 0;
  y[i] = 0;
  //if (i < n) {
    //y[i] = a * x[i] + y[i];
  //}
}

void saxpy_cuda(int n, real alpha, real *x, real *y) {
  real* d_x, *d_y;
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
  cudaMemcpy(x, d_x, n * sizeof(real), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  for (int i = 0; i < n; i++) {
    printf("%f %f\n", x[i], y[i]);
  }
  cudaFree(d_x);
  cudaFree(d_y);
  printf("done\n");
}
