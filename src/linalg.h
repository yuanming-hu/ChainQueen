#pragma once

#define TC_FORCE_INLINE __forceinline__
using real = float;

class Vector {
 public:
  static constexpr int dim = 3;

  real d[dim];

  TC_FORCE_INLINE __device__ Vector(real x, real y, real z) {
    d[0] = x;
    d[1] = y;
    d[2] = z;
  }

  TC_FORCE_INLINE __device__ Vector(real x = 0) {
    for (int i = 0; i < dim; i++) {
      d[i] = x;
    }
  }

  TC_FORCE_INLINE __device__ real operator[](int i) {
    return d[i];
  }
};

class Matrix {
 public:
  static constexpr int dim = 3;

  real d[dim][dim];

  TC_FORCE_INLINE __device__ Matrix(real a00,
         real a01,
         real a02,
         real a10,
         real a11,
         real a12,
         real a20,
         real a21,
         real a22) {
    d[0][0] = a00;
    d[0][1] = a01;
    d[0][2] = a02;
    d[1][0] = a10;
    d[1][1] = a11;
    d[1][2] = a12;
    d[2][0] = a20;
    d[2][1] = a21;
    d[2][2] = a22;
  }

  TC_FORCE_INLINE __device__ Matrix() {
    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
        d[i][j] = 0;
      }
    }
  }
};
