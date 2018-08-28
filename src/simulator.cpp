#include <taichi/common/util.h>
#include <taichi/common/task.h>
#include "kernels.h"

TC_NAMESPACE_BEGIN

class DMPMSimulator3D {
 public:
  DMPMSimulator3D() {
  }

  struct State {};

  void forward(const State &state) {
  }

  void backward(const State &initial_state, const State &new_state) {
  }

  void test() {
  }
};

auto test_cuda = []() {
  test();
  int N = 10;
  std::vector<real> a(N), b(N);
  for (int i = 0; i < N; i++) {
    a[i] = i;
    b[i] = i * 2;
  }
  saxpy_cuda(N, 2.5_f, a.data(), b.data());
  for (int i = 0; i < N; i++) {
    TC_ASSERT(b[i] == i * 4);
  }
};

TC_REGISTER_TASK(test_cuda);

TC_NAMESPACE_END
