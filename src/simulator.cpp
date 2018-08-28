#include <taichi/common/util.h>
#include <taichi/common/task.h>
#include <taichi/testing.h>
#include "kernels.h"
#include "particle.h"

TC_NAMESPACE_BEGIN

class DMPMSimulator3D {
 public:
  State state;

  DMPMSimulator3D() {
    // TODO: initialize the state
  }


  void advance() {
    ::advance(state);
  }

  void backward(const State &initial_state, const State &new_state) {
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

TC_NAMESPACE_END
