#include <taichi/common/util.h>
#include <taichi/common/task.h>
#include <taichi/testing.h>
#include "kernels.h"
#include "particle.h"

TC_NAMESPACE_BEGIN

class DMPMSimulator3D {
 public:
  StateBase state;

  DMPMSimulator3D() {
    // TODO: initialize the state
  }

  void advance() {
    ::advance(state);
  }

  void backward(const StateBase &initial_state, const StateBase &new_state) {
  }

  void test() {
  }
};

auto test_cuda = []() {
  int N = 10;
  std::vector<real> a(N), b(N);
  for (int i = 0; i < N; i++) {
    a[i] = i;
    b[i] = i * 2;
  }
  saxpy_cuda(N, 2.0_f, a.data(), b.data());
  for (int i = 0; i < N; i++) {
    TC_ASSERT_EQUAL(b[i], i * 4.0_f, 1e-5_f);
  }
};

TC_REGISTER_TASK(test_cuda);

auto test_cuda_svd = []() {
  int N = 12800;
  using Matrix = Matrix3f;
  std::vector<Matrix> A, U, sig, V;
  A.resize(N);
  U.resize(N);
  sig.resize(N);
  V.resize(N);

  std::vector<real> A_flattened;
  for (int p = 0; p < N; p++) {
    auto matA = Matrix(1) + 0.5_f * Matrix::rand();
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        A_flattened.push_back(matA(i, j));
      }
    }
    A[p] = matA;
  }

  test_svd_cuda(N, (real *)A_flattened.data(), (real *)U.data(),
                (real *)sig.data(), (real *)V.data());

  constexpr real tolerance = 3e-5_f32;
  for (int i = 0; i < N; i++) {
    auto matA = A[i];
    auto matU = U[i];
    auto matV = V[i];
    auto matSig = sig[i];

    TC_ASSERT_EQUAL(matSig, Matrix(matSig.diag()), tolerance);
    TC_ASSERT_EQUAL(Matrix(1), matU * transposed(matU), tolerance);
    TC_ASSERT_EQUAL(Matrix(1), matV * transposed(matV), tolerance);
    TC_ASSERT_EQUAL(matA, matU * matSig * transposed(matV), tolerance);

    /*
      polar_decomp(m, R, S);
      TC_CHECK_EQUAL(m, R * S, tolerance);
      TC_CHECK_EQUAL(Matrix(1), R * transposed(R), tolerance);
      TC_CHECK_EQUAL(S, transposed(S), tolerance);
     */
  }
};

TC_REGISTER_TASK(test_cuda_svd);

TC_NAMESPACE_END
