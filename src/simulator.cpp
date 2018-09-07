#include <taichi/common/util.h>
#include <taichi/common/task.h>
#include <taichi/testing.h>
#include <taichi/io/optix.h>
#include "kernels.h"
#include "state_base.h"

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

/*
auto gpu_mpm3d = []() {
  void *states;
  int num_particles = 30 * 30 * 30;
  std::vector<real> initial_positions;
  for (int i = 0; i < 30; i++) {
    for (int j = 0; j < 30; j++) {
      for (int k = 0; k < 30; k++) {
        initial_positions.push_back(i * 0.005_f + 0.4_f);
      }
    }
  }
  for (int i = 0; i < 30; i++) {
    for (int j = 0; j < 30; j++) {
      for (int k = 0; k < 30; k++) {
        initial_positions.push_back(j * 0.005_f + 0.6_f);
      }
    }
  }
  for (int i = 0; i < 30; i++) {
    for (int j = 0; j < 30; j++) {
      for (int k = 0; k < 30; k++) {
        initial_positions.push_back(k * 0.005_f + 0.4_f);
      }
    }
  }
  initialize_mpm3d_state(states, initial_positions.data());

  for (int i = 0; i < 150; i++) {
    auto x = fetch_mpm3d_particles(states);
    OptiXScene scene;
    for (int p = 0; p < 30 * 30 * 30; p++) {
      OptiXParticle particle;
      auto scale = 5_f;
      particle.position_and_radius =
          Vector4(x[p] * scale, (x[p + num_particles] - 0.02f) * scale,
                  x[p + 2 * num_particles] * scale, 0.01);
      if (p == 123)
        TC_P(particle.position_and_radius);
      scene.particles.push_back(particle);
    }
    write_to_binary_file(scene, fmt::format("{:05d}.tcb", i));
    for (int j = 0; j < 10; j++) {
      advance_mpm3d_state(states);
    }
  }
};
*/

auto gpu_mpm3d = []() {
  int n = 4;
  int num_particles = n * n * n;
  std::vector<real> initial_positions;
  std::vector<real> initial_velocities;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < n; k++) {
        bool right =  (i / (n / 2));
        initial_positions.push_back(i * 0.025_f + 0.2123_f + 0.1 * right);
        initial_velocities.push_back(1 - 1*right);
        // initial_velocities.push_back(0.1);
      }
    }
  }
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < n; k++) {
        initial_positions.push_back(j * 0.025_f + 0.4344_f);
        initial_velocities.push_back(0);
      }
    }
  }
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < n; k++) {
        initial_positions.push_back(k * 0.025_f + 0.9854_f);
        initial_velocities.push_back(0);
      }
    }
  }
  int num_steps = 30;
  std::vector<void *> states((uint32)num_steps + 1, nullptr);
  Vector3i res(20);
  // Differentiate gravity is not supported
  Vector3 gravity(0, 0, 0);
  for (int i = 0; i < num_steps + 1; i++) {
    initialize_mpm3d_state(&res[0], num_particles, &gravity[0], states[i],
                           1e-2_f, initial_positions.data());
    std::fill(initial_positions.begin(), initial_positions.end(), 0);
    if (i == 0) {
      set_initial_velocities(states[i], initial_velocities.data());
    }
  }

  for (int i = 0; i < num_steps; i++) {
    TC_INFO("forward step {}", i);
    auto x = fetch_mpm3d_particles(states[i]);
    OptiXScene scene;
    for (int p = 0; p < (int)initial_positions.size() / 3; p++) {
      OptiXParticle particle;
      auto scale = 5_f;
      particle.position_and_radius =
          Vector4(x[p] * scale, (x[p + num_particles] - 0.02f) * scale,
                  x[p + 2 * num_particles] * scale, 0.03);
      scene.particles.push_back(particle);
      if (p == 0)
        TC_P(particle.position_and_radius);
    }
    int interval = 3;
    if (i % interval == 0) {
      write_to_binary_file(scene, fmt::format("{:05d}.tcb", i / interval));
    }
    forward_mpm3d_state(states[i], states[i + 1]);
  }

  set_grad_loss(states[num_steps]);
  for (int i = num_steps - 1; i >= 0; i--) {
    TC_INFO("backward step {}", i);
    backward_mpm3d_state(states[i], states[i + 1]);
    auto grad_x = fetch_mpm3d_grad_x(states[i]);
    auto grad_v = fetch_mpm3d_grad_v(states[i]);
    Vector3f vgrad_v, vgrad_x;
    for (int j = 0; j < num_particles; j++) {
      for (int k = 0; k < 3; k++) {
        vgrad_v[k] += grad_v[k * num_particles + j];
        vgrad_x[k] += grad_x[k * num_particles + j];
      }
    }
    TC_P(vgrad_v);
    TC_P(vgrad_x);
  }
};

TC_REGISTER_TASK(gpu_mpm3d);

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
