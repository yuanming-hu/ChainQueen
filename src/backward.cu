#include "kernels.h"
#include "linalg.h"
#include "particle.h"
#include "svd.cuh"
#include <cstdio>
#include <vector>

// For deformation gradient update

// Takes B, dL/dAB
// Returns dL/dA
__device__ Matrix dAB2dA(const Matrix &B, const Matrix &dAB) {
  Matrix dA;
  for (int p = 0; p < dim; p++) {
    for (int q = 0; q < dim; q++) {
      for (int j = 0; j < dim; j++) {
        dA[p][q] += dAB[p][j] * B[q][j];
      }
    }
  }
  return dA;
}

// Takes A, B, dL/dAB
// Returns dL/dB
__device__ Matrix dAB2dB(const Matrix &A, const Matrix &dAB) {
  Matrix dB;
  for (int p = 0; p < dim; p++) {
    for (int q = 0; q < dim; q++) {
      for (int i = 0; i < dim; i++) {
        dB[p][q] += dAB[i][q] * A[i][p];
      }
    }
  }
  return dB;
}

__device__ Vector duTvdu(const Vector &v, const real &duTv) {
  return duTv * v;
}

__device__ Vector duTvdv(const Vector &u, const real &duTv) {
  return duTv * u;
}


