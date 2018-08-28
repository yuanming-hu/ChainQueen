#include "kernels.h"
#include "linalg.h"
#include "particle.h"

constexpr int dim = 3;
constexpr int spline_size = 3;

__global__ void saxpy(int n, real a, real *x, real *y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] = a * x[i] + y[i];
  }
}

void saxpy_cuda(int N, real alpha, real *x, real *y) {
  real *d_x, *d_y;

  cudaMalloc(&d_x, N * sizeof(real));
  cudaMalloc(&d_y, N * sizeof(real));

  cudaMemcpy(d_x, x, N * sizeof(real), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, N * sizeof(real), cudaMemcpyHostToDevice);

  // this is stupidly wrong..... saxpy_g<<<1, 256>>>(n, alpha, x, y);
  saxpy<<<(N + 255) / 256, 256>>>(N, alpha, d_x, d_y);

  cudaMemcpy(y, d_y, N * sizeof(real), cudaMemcpyDeviceToHost);

  cudaFree(d_x);
  cudaFree(d_y);
}

__global__ void P2G(State &state) {
  // One particle per thread

  auto inv_dx = real(1.0) / state.dx;

  constexpr int scratch_size = 8;
  __shared__ real scratch[dim + 1][scratch_size][scratch_size][scratch_size];

  // load from global memory
  real mass = 0;
  Vector x, v;

  real weight[dim][spline_size];
  // Compute B-Spline weights
  for (int v = 0; v < dim; ++v) {
    real d0 = x[v] * inv_dx;
    real z = ((real)1.5 - d0);
    weight[v][0] = (real)0.5 * z * z;
    d0 = d0 - 1.0f;
    weight[v][1] = (real)0.75 - d0 * d0;
    z = (real)1.5 - (1.0f - d0);
    weight[v][2] = (real)0.5 * z * z;
  }

  real val[dim + 1];
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 3; k++) {
        auto w = weight[0][i] * weight[1][j] * weight[2][k];

        val[0] = mass * weight;
        Vector dpos;

        // reduce in warp
        for (int iter = 1; iter <= mark; iter <<= 1) {
          T tmp[4];
          for (int i = 0; i < 4; ++i)
            tmp[i] = __shfl_down(val[i], iter);
          if (interval >= iter)
            for (int i = 0; i < 4; ++i)
              val[i] += tmp[i];
        }

        // cross-warp atomic
        for (int r = 0; r < dim + 1; r++) {
          atomicAdd();
        }
      }
    }
  }
}

void sort(State &state) {

}

__global__ void G2P(State &state) {
}

__global__ void normalize_grid(State &state) {
}

void advance(State &state) {
  sort(state);
  P2G<<<>>>(state);
  normalize_grid<<<>>>(state);
  G2P<<<>>>(state);
}
