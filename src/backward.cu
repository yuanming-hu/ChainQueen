#include "kernels.h"
#include "linalg.h"
#include "particle.h"
#include "svd.cuh"
#include <cstdio>
#include <vector>

struct Tensor4 {

};

__device__ Tensor4 dABdA(const Matrix &A, const Matrix &B, const Matrix &AB) {

}

__device__ Vector duTvBdu(const Vector &u) {
  return u;
}

// Row vector?
__device__ Vector duTvBdv(const Vector &u) {
  return v;
}

// Row vector?
__device__ Vector duTvBdv(const Vector &u) {
  return v;
}
