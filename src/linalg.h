#pragma once

#define TC_FORCE_INLINE __forceinline__
using real = float;
constexpr int dim = 3;

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

  TC_FORCE_INLINE __host__ __device__ real operator[](int i) const {
    return d[i];
  }

  TC_FORCE_INLINE __host__ __device__ real &operator[](int i) {
    return d[i];
  }

  __device__ Vector operator-(const Vector &o) {
    Vector ret;
    for (int i = 0; i < dim; i++) {
      ret[i] = d[i] - o[i];
    }
    return ret;
  }
};

__device__ Vector operator*(real alpha, const Vector &o) {
  Vector ret;
  for (int i = 0; i < dim; i++) {
    ret[i] = alpha * o[i];
  }
  return ret;
}

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

  TC_FORCE_INLINE __device__ Matrix(real x = 0) {
    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
        d[i][j] = x;
      }
    }
  }

  TC_FORCE_INLINE __host__ __device__ real *operator[](int i) {
    return d[i];
  }

  TC_FORCE_INLINE __host__ __device__ real const *operator[](int i) const {
    return d[i];
  }

  __device__ Matrix operator*(const Matrix &o) {
    Matrix ret;
    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
        for (int k = 0; k < dim; k++) {
          ret[i][j] += d[i][k] * o[k][j];
        }
      }
    }
    return ret;
  }

  __device__ Matrix operator+(const Matrix &o) {
    Matrix ret;
    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
        ret[i][j] = d[i][j] + o[i][j];
      }
    }
    return ret;
  }

  __device__ Matrix operator-(const Matrix &o) {
    Matrix ret;
    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
        ret[i][j] = d[i][j] - o[i][j];
      }
    }
    return ret;
  }

  __device__ Vector operator*(const Vector &v) {
    Vector ret;
    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
        ret[i] += d[i][j] * v[j];
      }
    }
    return ret;
  }
};

__device__ Matrix transposed(const Matrix &A) {
  return Matrix(A[0][0], A[1][0], A[2][0], A[0][1], A[1][1], A[2][1], A[0][2],
                A[1][2], A[2][2]);
}

__device__ TC_FORCE_INLINE real determinant(const Matrix &mat) {
  return mat[0][0] * (mat[1][1] * mat[2][2] - mat[2][1] * mat[1][2]) -
         mat[1][0] * (mat[0][1] * mat[2][2] - mat[2][1] * mat[0][2]) +
         mat[2][0] * (mat[0][1] * mat[1][2] - mat[1][1] * mat[0][2]);
}

__device__ Matrix operator*(real alpha, const Matrix &o) {
  Matrix ret;
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      ret[i][j] = alpha * o[i][j];
    }
  }
  return ret;
}

TC_FORCE_INLINE __device__ real sqr(real x) {
  return x * x;
}
