#include <taichi/common/util.h>
#include <taichi/common/task.h>
#include <taichi/testing.h>
#include <taichi/io/optix.h>
#include <Partio.h>
#include <taichi/system/profiler.h>
#include "config.h"
#include "kernels.h"
#include "state_base.h"

TC_NAMESPACE_BEGIN

void write_partio(std::vector<Vector3> positions,
                  const std::string &file_name) {
  Partio::ParticlesDataMutable *parts = Partio::create();
  Partio::ParticleAttribute posH, vH, mH, typeH, normH, statH, boundH, distH,
      debugH, indexH, limitH, apicH;

  bool verbose = false;

  posH = parts->addAttribute("position", Partio::VECTOR, 3);
  // typeH = parts->addAttribute("type", Partio::INT, 1);
  // indexH = parts->addAttribute("index", Partio::INT, 1);
  // limitH = parts->addAttribute("limit", Partio::INT, 3);
  // vH = parts->addAttribute("v", Partio::VECTOR, 3);

  if (verbose) {
    mH = parts->addAttribute("m", Partio::VECTOR, 1);
    normH = parts->addAttribute("boundary_normal", Partio::VECTOR, 3);
    debugH = parts->addAttribute("debug", Partio::VECTOR, 3);
    statH = parts->addAttribute("states", Partio::INT, 1);
    distH = parts->addAttribute("boundary_distance", Partio::FLOAT, 1);
    boundH = parts->addAttribute("near_boundary", Partio::INT, 1);
    apicH = parts->addAttribute("apic_frobenius_norm", Partio::FLOAT, 1);
  }
  for (auto p : positions) {
    // const Particle *p = allocator.get_const(p_i);
    int idx = parts->addParticle();
    // Vector vel = p->get_velocity();
    // float32 *v_p = parts->dataWrite<float32>(vH, idx);
    // for (int k = 0; k < 3; k++)
    //  v_p[k] = vel[k];
    // int *type_p = parts->dataWrite<int>(typeH, idx);
    // int *index_p = parts->dataWrite<int>(indexH, idx);
    // int *limit_p = parts->dataWrite<int>(limitH, idx);
    float32 *p_p = parts->dataWrite<float32>(posH, idx);

    // Vector pos = p->pos;

    for (int k = 0; k < 3; k++)
      p_p[k] = 0.f;

    for (int k = 0; k < 3; k++)
      p_p[k] = p[k];
    // type_p[0] = int(p->is_rigid());
    // index_p[0] = p->id;
    // limit_p[0] = p->dt_limit;
    // limit_p[1] = p->stiffness_limit;
    // limit_p[2] = p->cfl_limit;
  }
  Partio::write(file_name.c_str(), *parts);
  parts->release();
}

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
  constexpr int dim = 3;
  int n = 16;
  int num_particles = n * n * n;
  std::vector<real> initial_positions;
  std::vector<real> initial_velocities;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < n; k++) {
        bool right = (i / (n / 2));
        // initial_positions.push_back(i * 0.025_f + 0.2123_f);
        initial_positions.push_back(i * 0.025_f + 0.2123_f + 0.2 * right);
        initial_velocities.push_back(1 - 1 * right);
        // initial_velocities.push_back(0.0);
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
  std::vector<real> initial_F;
  for (int a = 0; a < 3; a++) {
    for (int b = 0; b < 3; b++) {
      for (int i = 0; i < num_particles; i++) {
        initial_F.push_back(real(a == b) * 1.1);
      }
    }
  }
  TC_P(initial_F.size());
  int num_steps = 80;
  std::vector<TStateBase<dim> *> states((uint32)num_steps + 1, nullptr);
  Vector3i res(20);
  // Differentiate gravity is not supported
  Vector3 gravity(0, 0, 0);
  for (int i = 0; i < num_steps + 1; i++) {
    initialize_mpm3d_state(&res[0], num_particles, &gravity[0],
                           (void *&)states[i], 1.0_f / res[0], 5e-3_f,
                           initial_positions.data());
    std::fill(initial_positions.begin(), initial_positions.end(), 0);
    if (i == 0) {
      states[i]->set_initial_v(initial_velocities.data());
      states[i]->set_initial_F(initial_F.data());
    }
  }

  for (int i = 0; i < num_steps; i++) {
    TC_INFO("forward step {}", i);
    auto x = states[i]->fetch_x();
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
    auto grad_x = states[i]->fetch_grad_x();
    auto grad_v = states[i]->fetch_grad_v();
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

auto gpu_mpm3d_falling_cube = []() {
  constexpr int dim = 3;
  // The cube has size 2 * 2 * 2, with height 5m, falling time = 1s, g=-10
  int n = 80;
  real dx = 0.2;
  real sample_density = 0.1;
  Vector3 corner(2, 5 + 2 * dx, 2);
  int num_particles = n * n * n;
  std::vector<real> initial_positions;
  std::vector<real> initial_velocities;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < n; k++) {
        // initial_positions.push_back(i * 0.025_f + 0.2123_f);
        initial_positions.push_back(i * sample_density + corner[0]);
        initial_velocities.push_back(0);
      }
    }
  }
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < n; k++) {
        initial_positions.push_back(j * sample_density + corner[1]);
        initial_velocities.push_back(0);
      }
    }
  }
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < n; k++) {
        initial_positions.push_back(k * sample_density + corner[2]);
        initial_velocities.push_back(0);
      }
    }
  }
  std::vector<real> initial_F;
  for (int a = 0; a < 3; a++) {
    for (int b = 0; b < 3; b++) {
      for (int i = 0; i < num_particles; i++) {
        initial_F.push_back(real(a == b) * 1.1);
      }
    }
  }
  int num_frames = 300;
  Vector3i res(100, 120, 100);
  Vector3 gravity(0, -10, 0);
  TStateBase<dim> *state;
  TStateBase<dim> *state2;
  int substep = 3;
  real dt = 1.0_f / 60 / substep;
  initialize_mpm3d_state(&res[0], num_particles, &gravity[0], (void *&)state, dx, dt,
                         initial_positions.data());
  reinterpret_cast<TStateBase<3> *>(state)->set(10, 100, 5000, 0.3);
  initialize_mpm3d_state(&res[0], num_particles, &gravity[0], (void *&)state2, dx, dt,
                         initial_positions.data());
  reinterpret_cast<TStateBase<3> *>(state2)->set(10, 100, 5000, 0.3);
  state->set_initial_v(initial_velocities.data());

  for (int i = 0; i < num_frames; i++) {
    TC_INFO("forward step {}", i);
    auto x = state->fetch_x();
    auto fn = fmt::format("{:04d}.bgeo", i);
    TC_INFO(fn);
    std::vector<Vector3> parts;
    for (int p = 0; p < (int)initial_positions.size() / 3; p++) {
      auto pos = Vector3(x[p], x[p + num_particles], x[p + 2 * num_particles]);
      parts.push_back(pos);
    }
    write_partio(parts, fn);

    {
      TC_PROFILER("simulate one frame");
      for (int j = 0; j < substep; j++)
        forward_mpm3d_state(state, state);
    }
    taichi::print_profile_info();
  }
  while (true) {
    TC_PROFILER("backward");
    for (int j = 0; j < substep; j++)
      backward_mpm3d_state(state2, state);
    taichi::print_profile_info();
  }
};

TC_REGISTER_TASK(gpu_mpm3d_falling_cube);

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

auto test_partio = []() {
  real dx = 0.01_f;
  for (int f = 0; f < 100; f++) {
    std::vector<Vector3> positions;
    for (int i = 0; i < 10; i++) {
      for (int j = 0; j < 10; j++) {
        for (int k = 0; k < 10; k++) {
          positions.push_back(dx * Vector3(i + f, j, k));
        }
      }
    }
    auto fn = fmt::format("{:04d}.bgeo", f);
    TC_INFO(fn);
    write_partio(positions, fn);
  }
};

TC_REGISTER_TASK(test_partio);

auto write_partio_c = [](const std::vector<std::string> &parameters) {
  auto n = (int)std::atoi(parameters[0].c_str());
  float *pos_ = reinterpret_cast<float *>(std::atol(parameters[1].c_str()));
  auto fn = parameters[2];
  using namespace taichi;
  std::vector<Vector3> pos;
  for (int i = 0; i < n; i++) {
    auto p = Vector3(pos_[i], pos_[i + n], pos_[i + 2 * n]);
    pos.push_back(p);
  }
  taichi::write_partio(pos, fn);
};

TC_REGISTER_TASK(write_partio_c);

TC_NAMESPACE_END
