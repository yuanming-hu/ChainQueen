import unittest
from simulation import Simulation
from time_integration import UpdatedSimulationState
import tensorflow as tf
import numpy as np

sess = tf.Session()


class TestSimulator(unittest.TestCase):

  def assertAlmostEqualFloat32(self, a, b):
    if abs(a - b) > 1e-5 * max(a, b):
      self.assertEqual(a, b)

  def test_acceleration(self):
    pass

  def motion_test(self,
                  gravity=(0, -10),
                  initial_velocity=(0, 0),
                  batch_size=1,
                  dx=1,
                  num_steps=10):
    # Zero gravity, 1-batched, translating block
    num_particles = 100
    sim = Simulation(
        sess=sess,
        grid_res=(30, 30),
        dx=dx,
        num_particles=num_particles,
        gravity=gravity,
        batch_size=batch_size)
    initial = sim.initial_state
    next_state = UpdatedSimulationState(sim, initial)
    position = np.zeros(shape=(batch_size, num_particles, 2))
    velocity = np.zeros(shape=(batch_size, num_particles, 2))
    for b in range(batch_size):
      for i in range(10):
        for j in range(10):
          position[b, i * 10 + j] = ((i * 0.5 + 12.75) * dx,
                                     (j * 0.5 + 12.75) * dx)
          velocity[b, i * 10 + j] = initial_velocity
    input_state = sim.get_initial_state(position=position, velocity=velocity)

    def center_of_mass():
      return np.mean(input_state[0][:, :, 0]), np.mean(input_state[0][:, :, 1])

    x, y = 15.0 * dx, 15.0 * dx
    vx, vy = initial_velocity

    self.assertAlmostEqual(center_of_mass()[0], x)
    self.assertAlmostEqual(center_of_mass()[1], y)
    for i in range(num_steps):
      input_state = sess.run(
          next_state.to_tuples(),
          feed_dict={
              sim.initial_state_place_holder(): input_state
          })

      # This will work if we use Verlet
      # self.assertAlmostEqual(center_of_mass()[1], 15.0 - t * t * 0.5 * g)

      # Symplectic Euler version
      vx += sim.dt * gravity[0]
      x += sim.dt * vx
      vy += sim.dt * gravity[1]
      y += sim.dt * vy
      self.assertAlmostEqualFloat32(center_of_mass()[0], x)
      self.assertAlmostEqualFloat32(center_of_mass()[1], y)

  def test_translation_x(self):
    self.motion_test(initial_velocity=(1, 0))

  def test_translation_x_batched(self):
    self.motion_test(initial_velocity=(1, 0), batch_size=2)

  def test_translation_y(self):
    self.motion_test(initial_velocity=(0, 1))

  def test_falling_translation(self):
    self.motion_test(initial_velocity=(2, -1), gravity=(-4, 6))

  def test_falling_translation_dx(self):
    self.motion_test(initial_velocity=(2, -1), gravity=(-4, 6), dx=0.05)
    self.motion_test(
        initial_velocity=(0.02, -0.01), gravity=(-0.04, 0.06), dx=0.1)
    self.motion_test(initial_velocity=(2, -1), gravity=(-4, 6), dx=10)

  def test_free_fall(self):
    self.motion_test(gravity=(0, -10))

  def test_recursive_placeholder(self):
    a = tf.placeholder(dtype=tf.float32)
    b = tf.placeholder(dtype=tf.float32)
    self.assertAlmostEqual(sess.run(a + b, feed_dict={(a, b): [1, 2]}), 3)
    # The following will not work
    # print(sess.run(a + b, feed_dict={{'a':a, 'b':b}: {'a':1, 'b':2}}))

  def test_bouncing_cube(self):
    gravity = (0, -10)
    batch_size = 2
    dx = 0.03
    num_particles = 100
    sim = Simulation(
        grid_res=(30, 30),
        dx=dx,
        num_particles=num_particles,
        gravity=gravity,
        dt=1e-3,
        batch_size=batch_size,
        sess=sess)
    position = np.zeros(shape=(batch_size, num_particles, 2))
    poissons_ratio = np.ones(shape=(batch_size, num_particles, 1)) * 0.45
    initial_velocity = tf.placeholder(shape=(2,), dtype=tf.float32)
    velocity = tf.broadcast_to(initial_velocity[None, None, :], shape=(batch_size, num_particles, 2))
    for b in range(batch_size):
      for i in range(10):
        for j in range(10):
          position[b, i * 10 + j] = (((i + b * 3) * 0.5 + 12.75) * dx,
                                     (j * 0.5 + 12.75) * dx)
    input_state = sim.get_initial_state(
        position=position, velocity=velocity, poissons_ratio=poissons_ratio)

    frames = sim.run(input_state, 1000, initial_feed_dict={
      initial_velocity: [1, 0]
    })
    for i in range(100):
      sim.visualize_particles(frames[i * 10][0][1])

  def test_rotating_cube(self):
    gravity = (0, 0)
    batch_size = 1
    dx = 0.03
    num_particles = 100
    sim = Simulation(
        grid_res=(30, 30),
        dx=dx,
        num_particles=num_particles,
        gravity=gravity,
        dt=1e-3,
        sess=sess)
    position = np.zeros(shape=(batch_size, num_particles, 2))
    velocity = np.zeros(shape=(batch_size, num_particles, 2))
    for b in range(batch_size):
      for i in range(10):
        for j in range(10):
          position[b, i * 10 + j] = ((i * 0.5 + 12.75) * dx,
                                     (j * 0.5 + 12.75) * dx)
          velocity[b, i * 10 + j] = (1 * (j - 4.5), -1 * (i - 4.5))
    input_state = sim.get_initial_state(position=position, velocity=velocity)

    memo = sim.run(input_state, 100)
    for i in range(100):
      sim.visualize_particles(memo.steps[i][0][0])

  def test_dilating_cube(self):
    gravity = (0, 0)
    batch_size = 1
    dx = 0.03
    num_particles = 100
    sim = Simulation(
        grid_res=(30, 30),
        dx=dx,
        num_particles=num_particles,
        gravity=gravity,
        dt=1e-3,
        sess=sess)
    position = np.zeros(shape=(batch_size, num_particles, 2))
    velocity = np.zeros(shape=(batch_size, num_particles, 2))
    youngs_modulus = np.zeros(shape=(batch_size, num_particles, 1))
    for b in range(batch_size):
      for i in range(10):
        for j in range(10):
          position[b, i * 10 + j] = ((i * 0.5 + 12.75) * dx,
                                     (j * 0.5 + 12.75) * dx)
          velocity[b, i * 10 + j] = (0.5 * (i - 4.5), 0)
    input_state = sim.get_initial_state(
        position=position, velocity=velocity, youngs_modulus=youngs_modulus)

    memo = sim.run(input_state, 100)
    for i in range(100):
      sim.visualize_particles(memo.steps[i][0][0])
      
  def test_sess(self):
    print(sess.run(1))

  def test_initial_gradient(self):
    gravity = (0, 0)
    batch_size = 1
    dx = 0.03
    num_particles = 100
    sim = Simulation(
      grid_res=(30, 30),
      dx=dx,
      num_particles=num_particles,
      gravity=gravity,
      dt=1e-3,
      sess=sess)
    position = np.zeros(shape=(batch_size, num_particles, 2))
    youngs_modulus = np.zeros(shape=(batch_size, num_particles, 1))
    velocity_ph = tf.placeholder(shape=(2,), dtype=tf.float32)
    velocity = tf.broadcast_to(velocity_ph[None, None, :], [batch_size, num_particles, 2])
    for b in range(batch_size):
      for i in range(10):
        for j in range(10):
          position[b, i * 10 + j] = ((i * 0.5 + 12.75) * dx,
                                     (j * 0.5 + 12.75) * dx)
    input_state = sim.get_initial_state(
      position=position, velocity=velocity, youngs_modulus=youngs_modulus)

    loss = tf.reduce_mean(sim.initial_state.center_of_mass()[:, 0])
    frames = sim.run(input_state, 10, initial_feed_dict={velocity_ph: [3, 2]})
    grad = sim.gradients(loss, frames, [velocity_ph])
    print(grad)
    #for i in range(100):
    #  sim.visualize_particles(frames[i][0][0])


if __name__ == '__main__':
  unittest.main()
