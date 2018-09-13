#pragma once

#include "svd.cuh"
#include <cstdio>

#define TC_FORCE_INLINE __forceinline__
using real = float;

template <typename T, int dim_>
class TVector {
 public:
  static constexpr int dim = dim_;

  T d[dim];

  TC_FORCE_INLINE __device__ TVector(T *val) {
    for (int i = 0; i < dim; i++) {
      d[i] = val[i];
    }
  }

  TC_FORCE_INLINE __device__ __host__ T *data() {
    return &d[0];
  }

  template <int dim__ = dim_>
  TC_FORCE_INLINE __device__ TVector(T x, T y) {
    static_assert(dim__ == 2, "");
    d[0] = x;
    d[1] = y;
  }

  template <int dim__ = dim_>
  TC_FORCE_INLINE __device__ TVector(T x, T y, T z) {
    static_assert(dim__ == 3, "");
    d[0] = x;
    d[1] = y;
    d[2] = z;
  }

  TC_FORCE_INLINE __device__ TVector(T x = 0) {
    for (int i = 0; i < dim; i++) {
      d[i] = x;
    }
  }

  TC_FORCE_INLINE __host__ __device__ T operator[](int i) const {
    return d[i];
  }

  TC_FORCE_INLINE __host__ __device__ T &operator[](int i) {
    return d[i];
  }

  TC_FORCE_INLINE __device__ TVector &operator+=(const TVector &o) {
    for (int i = 0; i < dim; i++) {
      d[i] += o[i];
    }
    return *this;
  }

  TC_FORCE_INLINE __device__ TVector operator+(const TVector &o) {
    TVector ret;
    for (int i = 0; i < dim; i++) {
      ret[i] = d[i] + o[i];
    }
    return ret;
  }

  TC_FORCE_INLINE __device__ TVector operator-(const TVector &o) {
    TVector ret;
    for (int i = 0; i < dim; i++) {
      ret[i] = d[i] - o[i];
    }
    return ret;
  }

  __device__ T length2() const {
    T ret = 0;
    for (int i = 0; i < dim; i++) {
      ret += d[i] * d[i];
    }
    return ret;
  }

  __device__ T dot(TVector &other) const {
    T ret = 0;
    for (int i = 0; i < dim; i++) {
      ret += d[i] * other[i];
    }
    return ret;
  }
};

template <typename T, int dim>
TC_FORCE_INLINE __device__ TVector<T, dim> operator*(real alpha,
                                                     const TVector<T, dim> &o) {
  TVector<T, dim> ret;
  for (int i = 0; i < dim; i++) {
    ret[i] = alpha * o[i];
  }
  return ret;
}

template <typename T, int dim_>
class TMatrix {
 public:
  static constexpr int dim = dim_;
  using Vector = TVector<T, dim>;

  real d[dim][dim];

  TC_FORCE_INLINE __device__ __host__ real *data() {
    return &d[0][0];
  }

  template <int dim__ = dim_>
  TC_FORCE_INLINE __device__ TMatrix(T a00, T a01, T a10, T a11) {
    static_assert(dim__ == 2, "");
    d[0][0] = a00;
    d[0][1] = a01;
    d[1][0] = a10;
    d[1][1] = a11;
  }

  template <int dim__ = dim_>
  TC_FORCE_INLINE __device__
  TMatrix(T a00, T a01, T a02, T a10, T a11, T a12, T a20, T a21, T a22) {
    static_assert(dim__ == 3, "");
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

  TC_FORCE_INLINE __host__ __device__ TMatrix(real x = 0) {
    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
        d[i][j] = (i == j) ? x : 0;
      }
    }
  }

  TC_FORCE_INLINE __host__ __device__ real *operator[](int i) {
    return d[i];
  }

  TC_FORCE_INLINE __host__ __device__ real const *operator[](int i) const {
    return d[i];
  }

  TC_FORCE_INLINE __device__ TMatrix operator*(const TMatrix &o) {
    TMatrix ret;
    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
        for (int k = 0; k < dim; k++) {
          ret[i][j] += d[i][k] * o[k][j];
        }
      }
    }
    return ret;
  }

  TC_FORCE_INLINE __device__ TMatrix operator+(const TMatrix &o) {
    TMatrix ret;
    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
        ret[i][j] = d[i][j] + o[i][j];
      }
    }
    return ret;
  }

  TC_FORCE_INLINE __device__ TMatrix operator-(const TMatrix &o) {
    TMatrix ret;
    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
        ret[i][j] = d[i][j] - o[i][j];
      }
    }
    return ret;
  }

  TC_FORCE_INLINE __device__ Vector operator*(const Vector &v) {
    Vector ret;
    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
        ret[i] += d[i][j] * v[j];
      }
    }
    return ret;
  }

  TC_FORCE_INLINE __device__ T &operator()(int i, int j) {
    return d[i][j];
  }

  TC_FORCE_INLINE __device__ const T &operator()(int i, int j) const {
    return d[i][j];
  }

  static __device__ TMatrix outer_product(const Vector &col,
                                          const Vector &row) {
    TMatrix ret;
    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
        ret[i][j] = col[i] * row[j];
      }
    }
    return ret;
  }

  TC_FORCE_INLINE __device__ TMatrix elementwise_dot(TMatrix o) {
    T ret = 0;
    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
        ret += (*this)[i][j] * o[i][j];
      }
    }
    return ret;
  }
};

using Matrix3 = TMatrix<real, 3>;

template <typename T>
TC_FORCE_INLINE __device__ TMatrix<T, 2> transposed(const TMatrix<T, 2> &A) {
  return TMatrix<T, 2>(A[0][0], A[1][0], A[0][1], A[1][1]);
}

template <typename T>
TC_FORCE_INLINE __device__ TMatrix<T, 3> transposed(const TMatrix<T, 3> &A) {
  return TMatrix<T, 3>(A[0][0], A[1][0], A[2][0], A[0][1], A[1][1], A[2][1],
                       A[0][2], A[1][2], A[2][2]);
}

template <typename T>
TC_FORCE_INLINE __device__ T determinant(const TMatrix<T, 2> &mat) {
  return mat[0][0] * mat[1][1] - mat[1][0] * mat[0][1];
}

template <typename T>
TC_FORCE_INLINE __device__ T determinant(const TMatrix<T, 3> &mat) {
  return mat[0][0] * (mat[1][1] * mat[2][2] - mat[2][1] * mat[1][2]) -
         mat[1][0] * (mat[0][1] * mat[2][2] - mat[2][1] * mat[0][2]) +
         mat[2][0] * (mat[0][1] * mat[1][2] - mat[1][1] * mat[0][2]);
}

template <typename T, int dim>
TC_FORCE_INLINE __device__ TMatrix<T, dim> operator*(T alpha,
                                                     const TMatrix<T, dim> &o) {
  TMatrix<T, dim> ret;
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

TC_FORCE_INLINE __device__ void svd(Matrix3 &A,
                                    Matrix3 &U,
                                    Matrix3 &sig,
                                    Matrix3 &V) {
  // clang-format off
  sig[0][1] = sig[0][2] = sig[1][0] = sig[1][2] = sig[2][0] = sig[2][1] = 0;
  svd(
      A[0][0], A[0][1], A[0][2],
      A[1][0], A[1][1], A[1][2],
      A[2][0], A[2][1], A[2][2],
      U[0][0], U[0][1], U[0][2],
      U[1][0], U[1][1], U[1][2],
      U[2][0], U[2][1], U[2][2],
      sig[0][0], sig[1][1], sig[2][2],
      V[0][0], V[0][1], V[0][2],
      V[1][0], V[1][1], V[1][2],
      V[2][0], V[2][1], V[2][2]
  );
  // clang-format on
}

TC_FORCE_INLINE void __device__ polar_decomp(TMatrix<real, 2> &m,
                                             TMatrix<real, 2> &R,
                                             TMatrix<real, 2> &S) {
  auto x = m(0, 0) + m(1, 1);
  auto y = m(1, 0) - m(0, 1);
  auto scale = 1.0f / sqrtf(x * x + y * y);
  auto c = x * scale;
  auto s = y * scale;
  R = TMatrix<real, 2>(c, -s, s, c);
  S = transposed(R) * m;
}

TC_FORCE_INLINE __device__ void polar_decomp(TMatrix<real, 3> &A,
                                             TMatrix<real, 3> &R,
                                             TMatrix<real, 3> &S) {
  TMatrix<real, 3> U, sig, V;
  svd(A, U, sig, V);
  R = U * transposed(V);
  S = V * sig * transposed(V);
}

TC_FORCE_INLINE __device__ real Clamp_Small_Magnitude(const real input) {
  real magnitude = input > 0 ? input : -input;
  real sign = input > 0 ? 1.f : -1.f;
  real output = magnitude > 1e-6 ? magnitude : 1e-6;
  return output * sign;
}

template <int dim>
__device__ void Times_Rotated_dP_dF_FixedCorotated(const real mu,
                                                   const real lambda,
                                                   TMatrix<real, dim> &F_,
                                                   TMatrix<real, dim> &dF_,
                                                   TMatrix<real, dim> &dP_);

TC_FORCE_INLINE __device__ TMatrix<real, 2> dR_from_dF(TMatrix<real, 2> &F,
                                                       TMatrix<real, 2> &R,
                                                       TMatrix<real, 2> &S,
                                                       TMatrix<real, 2> &dF) {
  using Matrix = TMatrix<real, 2>;
  using Vector = TVector<real, 2>;
  // set W = R^T dR = [  0    x  ]
  //                  [  -x   0  ]
  //
  // R^T dF - dF^T R = WS + SW
  //
  // WS + SW = [ x(s21 - s12)   x(s11 + s22) ]
  //           [ -x[s11 + s22]  x(s21 - s12) ]
  // ----------------------------------------------------
  Matrix lhs = transposed(R) * dF - transposed(dF) * R;
  real x = lhs(0, 1) / (S(0, 0) + S(1, 1));
  Matrix W = Matrix(0, x, -x, 0);
  return R * W;
}

template <>
inline void __device__
Times_Rotated_dP_dF_FixedCorotated<2>(real mu,
                                      real lambda,
                                      TMatrix<real, 2> &F,
                                      TMatrix<real, 2> &dF,
                                      TMatrix<real, 2> &dP) {
  using Matrix = TMatrix<real, 2>;
  using Vector = TVector<real, 2>;

  const auto j = determinant(F);
  Matrix r, s;
  polar_decomp(F, r, s);
  Matrix dR = dR_from_dF(F, r, s, dF);
  Matrix JFmT = Matrix(F(1, 1), -F(1, 0), -F(0, 1), F(0, 0));
  Matrix dJFmT = Matrix(dF(1, 1), -dF(1, 0), -dF(0, 1), dF(0, 0));
  dP = 2.0f * mu * (dF - dR) + lambda * JFmT.elementwise_dot(dF) * JFmT +
       lambda * (j - 1) * dJFmT;
}

template <>
inline __device__ void Times_Rotated_dP_dF_FixedCorotated<3>(
    const real mu,
    const real lambda,
    TMatrix<real, 3> &F_,
    TMatrix<real, 3> &dF_,
    TMatrix<real, 3> &dP_) {
  real *F = F_.data();
  real *dF = dF_.data();
  real *dP = dP_.data();
  real U[9];
  real S[3];
  real V[9];
  svd(F[0], F[3], F[6], F[1], F[4], F[7], F[2], F[5], F[8], U[0], U[3], U[6],
      U[1], U[4], U[7], U[2], U[5], U[8], S[0], S[1], S[2], V[0], V[3], V[6],
      V[1], V[4], V[7], V[2], V[5], V[8]);

  //
  real J = S[0] * S[1] * S[2];
  real scaled_mu = 2.f * mu;
  real scaled_lambda = lambda * (J - 1.f);
  real P_hat[3];
  P_hat[0] = scaled_mu * (S[0] - 1.f) + scaled_lambda * (S[1] * S[2]);
  P_hat[1] = scaled_mu * (S[1] - 1.f) + scaled_lambda * (S[0] * S[2]);
  P_hat[2] = scaled_mu * (S[2] - 1.f) + scaled_lambda * (S[0] * S[1]);

  real dP_hat_dSigma_upper[6];
  scaled_lambda = lambda * (2.f * J - 1.f) * J;
  for (int i = 0; i < 3; ++i)
    dP_hat_dSigma_upper[i] = scaled_mu + lambda * J * J / (S[i] * S[i]);
  dP_hat_dSigma_upper[3] = scaled_lambda / (S[0] * S[1]);
  dP_hat_dSigma_upper[4] = scaled_lambda / (S[0] * S[2]);
  dP_hat_dSigma_upper[5] = scaled_lambda / (S[1] * S[2]);

  scaled_lambda = -lambda * (J - 1.f) * J;
  real M[3];
  M[0] = 0.5f * (2.f * mu + scaled_lambda / (S[0] * S[1]));
  M[1] = 0.5f * (2.f * mu + scaled_lambda / (S[0] * S[2]));
  M[2] = 0.5f * (2.f * mu + scaled_lambda / (S[2] * S[2]));
  //

  real P[3];
  P[0] = 0.5f * (P_hat[0] + P_hat[1]) / Clamp_Small_Magnitude(S[0] + S[1]);
  P[1] = 0.5f * (P_hat[0] + P_hat[2]) / Clamp_Small_Magnitude(S[0] + S[2]);
  P[2] = 0.5f * (P_hat[1] + P_hat[2]) / Clamp_Small_Magnitude(S[1] + S[2]);

  real dF_hat[9];
  dF_hat[0] = (dF[0] * U[0] + dF[1] * U[1] + dF[2] * U[2]) * V[0] +
              (dF[3] * U[0] + dF[4] * U[1] + dF[5] * U[2]) * V[1] +
              (dF[6] * U[0] + dF[7] * U[1] + dF[8] * U[2]) * V[2];
  dF_hat[1] = (dF[0] * U[3] + dF[1] * U[4] + dF[2] * U[5]) * V[0] +
              (dF[3] * U[3] + dF[4] * U[4] + dF[5] * U[5]) * V[1] +
              (dF[6] * U[3] + dF[7] * U[4] + dF[8] * U[5]) * V[2];
  dF_hat[2] = (dF[0] * U[6] + dF[1] * U[7] + dF[2] * U[8]) * V[0] +
              (dF[3] * U[6] + dF[4] * U[7] + dF[5] * U[8]) * V[1] +
              (dF[6] * U[6] + dF[7] * U[7] + dF[8] * U[8]) * V[2];
  dF_hat[3] = (dF[0] * U[0] + dF[1] * U[1] + dF[2] * U[2]) * V[3] +
              (dF[3] * U[0] + dF[4] * U[1] + dF[5] * U[2]) * V[4] +
              (dF[6] * U[0] + dF[7] * U[1] + dF[8] * U[2]) * V[5];
  dF_hat[4] = (dF[0] * U[3] + dF[1] * U[4] + dF[2] * U[5]) * V[3] +
              (dF[3] * U[3] + dF[4] * U[4] + dF[5] * U[5]) * V[4] +
              (dF[6] * U[3] + dF[7] * U[4] + dF[8] * U[5]) * V[5];
  dF_hat[5] = (dF[0] * U[6] + dF[1] * U[7] + dF[2] * U[8]) * V[3] +
              (dF[3] * U[6] + dF[4] * U[7] + dF[5] * U[8]) * V[4] +
              (dF[6] * U[6] + dF[7] * U[7] + dF[8] * U[8]) * V[5];
  dF_hat[6] = (dF[0] * U[0] + dF[1] * U[1] + dF[2] * U[2]) * V[6] +
              (dF[3] * U[0] + dF[4] * U[1] + dF[5] * U[2]) * V[7] +
              (dF[6] * U[0] + dF[7] * U[1] + dF[8] * U[2]) * V[8];
  dF_hat[7] = (dF[0] * U[3] + dF[1] * U[4] + dF[2] * U[5]) * V[6] +
              (dF[3] * U[3] + dF[4] * U[4] + dF[5] * U[5]) * V[7] +
              (dF[6] * U[3] + dF[7] * U[4] + dF[8] * U[5]) * V[8];
  dF_hat[8] = (dF[0] * U[6] + dF[1] * U[7] + dF[2] * U[8]) * V[6] +
              (dF[3] * U[6] + dF[4] * U[7] + dF[5] * U[8]) * V[7] +
              (dF[6] * U[6] + dF[7] * U[7] + dF[8] * U[8]) * V[8];

  real dP_hat[9];
  dP_hat[0] = dP_hat_dSigma_upper[0] * dF_hat[0] +
              dP_hat_dSigma_upper[3] * dF_hat[4] +
              dP_hat_dSigma_upper[4] * dF_hat[8];
  dP_hat[4] = dP_hat_dSigma_upper[3] * dF_hat[0] +
              dP_hat_dSigma_upper[1] * dF_hat[4] +
              dP_hat_dSigma_upper[5] * dF_hat[8];
  dP_hat[8] = dP_hat_dSigma_upper[4] * dF_hat[0] +
              dP_hat_dSigma_upper[5] * dF_hat[4] +
              dP_hat_dSigma_upper[2] * dF_hat[8];
  dP_hat[3] = ((M[0] + P[0]) * dF_hat[3] + (M[0] - P[0]) * dF_hat[1]);
  dP_hat[1] = ((M[0] - P[0]) * dF_hat[3] + (M[0] + P[0]) * dF_hat[1]);
  dP_hat[6] = ((M[1] + P[1]) * dF_hat[6] + (M[1] - P[1]) * dF_hat[2]);
  dP_hat[2] = ((M[1] - P[1]) * dF_hat[6] + (M[1] + P[1]) * dF_hat[2]);
  dP_hat[7] = ((M[2] + P[2]) * dF_hat[7] + (M[2] - P[2]) * dF_hat[5]);
  dP_hat[5] = ((M[2] - P[2]) * dF_hat[7] + (M[2] + P[2]) * dF_hat[5]);

  dP[0] = (dP_hat[0] * U[0] + dP_hat[1] * U[3] + dP_hat[2] * U[6]) * V[0] +
          (dP_hat[3] * U[0] + dP_hat[4] * U[3] + dP_hat[5] * U[6]) * V[3] +
          (dP_hat[6] * U[0] + dP_hat[7] * U[3] + dP_hat[8] * U[6]) * V[6];
  dP[1] = (dP_hat[0] * U[1] + dP_hat[1] * U[4] + dP_hat[2] * U[7]) * V[0] +
          (dP_hat[3] * U[1] + dP_hat[4] * U[4] + dP_hat[5] * U[7]) * V[3] +
          (dP_hat[6] * U[1] + dP_hat[7] * U[4] + dP_hat[8] * U[7]) * V[6];
  dP[2] = (dP_hat[0] * U[2] + dP_hat[1] * U[5] + dP_hat[2] * U[8]) * V[0] +
          (dP_hat[3] * U[2] + dP_hat[4] * U[5] + dP_hat[5] * U[8]) * V[3] +
          (dP_hat[6] * U[2] + dP_hat[7] * U[5] + dP_hat[8] * U[8]) * V[6];
  dP[3] = (dP_hat[0] * U[0] + dP_hat[1] * U[3] + dP_hat[2] * U[6]) * V[1] +
          (dP_hat[3] * U[0] + dP_hat[4] * U[3] + dP_hat[5] * U[6]) * V[4] +
          (dP_hat[6] * U[0] + dP_hat[7] * U[3] + dP_hat[8] * U[6]) * V[7];
  dP[4] = (dP_hat[0] * U[1] + dP_hat[1] * U[4] + dP_hat[2] * U[7]) * V[1] +
          (dP_hat[3] * U[1] + dP_hat[4] * U[4] + dP_hat[5] * U[7]) * V[4] +
          (dP_hat[6] * U[1] + dP_hat[7] * U[4] + dP_hat[8] * U[7]) * V[7];
  dP[5] = (dP_hat[0] * U[2] + dP_hat[1] * U[5] + dP_hat[2] * U[8]) * V[1] +
          (dP_hat[3] * U[2] + dP_hat[4] * U[5] + dP_hat[5] * U[8]) * V[4] +
          (dP_hat[6] * U[2] + dP_hat[7] * U[5] + dP_hat[8] * U[8]) * V[7];
  dP[6] = (dP_hat[0] * U[0] + dP_hat[1] * U[3] + dP_hat[2] * U[6]) * V[2] +
          (dP_hat[3] * U[0] + dP_hat[4] * U[3] + dP_hat[5] * U[6]) * V[5] +
          (dP_hat[6] * U[0] + dP_hat[7] * U[3] + dP_hat[8] * U[6]) * V[8];
  dP[7] = (dP_hat[0] * U[1] + dP_hat[1] * U[4] + dP_hat[2] * U[7]) * V[2] +
          (dP_hat[3] * U[1] + dP_hat[4] * U[4] + dP_hat[5] * U[7]) * V[5] +
          (dP_hat[6] * U[1] + dP_hat[7] * U[4] + dP_hat[8] * U[7]) * V[8];
  dP[8] = (dP_hat[0] * U[2] + dP_hat[1] * U[5] + dP_hat[2] * U[8]) * V[2] +
          (dP_hat[3] * U[2] + dP_hat[4] * U[5] + dP_hat[5] * U[8]) * V[5] +
          (dP_hat[6] * U[2] + dP_hat[7] * U[5] + dP_hat[8] * U[8]) * V[8];
};

TC_FORCE_INLINE __device__ TMatrix<real, 2> inversed(
    const TMatrix<real, 2> &mat) {
  real det = determinant(mat);
  return (1 / det) *
         TMatrix<real, 2>(mat[1][1], -mat[0][1], -mat[1][0], mat[0][0]);
}

TC_FORCE_INLINE __device__ TMatrix<real, 3> inversed(
    const TMatrix<real, 3> &mat) {
  real det = determinant(mat);
  return 1.0f / det *
         TMatrix<real, 3>(mat[1][1] * mat[2][2] - mat[2][1] * mat[1][2],
                          mat[2][1] * mat[0][2] - mat[0][1] * mat[2][2],
                          mat[0][1] * mat[1][2] - mat[1][1] * mat[0][2],
                          mat[2][0] * mat[1][2] - mat[1][0] * mat[2][2],
                          mat[0][0] * mat[2][2] - mat[2][0] * mat[0][2],
                          mat[1][0] * mat[0][2] - mat[0][0] * mat[1][2],
                          mat[1][0] * mat[2][1] - mat[2][0] * mat[1][1],
                          mat[2][0] * mat[0][1] - mat[0][0] * mat[2][1],
                          mat[0][0] * mat[1][1] - mat[1][0] * mat[0][1]);
}

template <int dim>
TC_FORCE_INLINE __device__ TMatrix<real, dim> PK1(real mu,
                                                  real lambda,
                                                  TMatrix<real, dim> F) {
  real J = determinant(F);
  TMatrix<real, dim> r, s;
  polar_decomp(F, r, s);
  return 2 * mu * (F - r) +
         TMatrix<real, dim>(lambda * (J - 1) * J) * transposed(inversed(F));
}

template <int dim>
TC_FORCE_INLINE __device__ TMatrix<real, dim>
kirchhoff_stress(real mu, real lambda, TMatrix<real, dim> F) {
  real J = determinant(F);
  TMatrix<real, dim> r, s;
  polar_decomp(F, r, s);
  return 2 * mu * (F - r) * transposed(F) +
         TMatrix<real, dim>(lambda * (J - 1) * J);
}

TC_FORCE_INLINE __device__ real sgn(real x) {
  return x > 0 ? 1 : -1;
}


/*
// Takes B, dL/dAB
// Returns dL/dA
TC_FORCE_INLINE __device__ Matrix dAB2dA(const Matrix &B, const Matrix &dAB) {
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
TC_FORCE_INLINE __device__ Matrix dAB2dB(const Matrix &A, const Matrix &dAB) {
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

TC_FORCE_INLINE __device__ Vector duTv2du(const Vector &v, const real &duTv) {
  return duTv * v;
}

TC_FORCE_INLINE __device__ Vector duTv2dv(const Vector &u, const real &duTv) {
  return duTv * u;
}
*/
