import unittest
from simulation import Simulation
from time_integration import SimulationState, InitialSimulationState, UpdatedSimulationState
import tensorflow as tf
import numpy as np

sess = tf.Session()

class TestSimulator(unittest.TestCase):

  def test_acceleration(self):
    pass

  def test_free_translation(self):
    pass

  def test_recursive_placeholder(self):
    a = tf.placeholder(dtype=tf.float32)
    b = tf.placeholder(dtype=tf.float32)
    print(sess.run(a + b, feed_dict={(a, b): [1, 2]}))
    # The following will not work
    # print(sess.run(a + b, feed_dict={{'a':a, 'b':b}: {'a':1, 'b':2}}))

  def test_free_fall(self):
    # Zero gravity, 1-batched, translating block
    num_particles = 100
    sim = Simulation(grid_res=(30, 30), num_particles=num_particles)
    initial = sim.initial_state
    next_state = UpdatedSimulationState(sim, initial)
    position = np.zeros(shape=(1, num_particles, 2))
    for i in range(10):
      for j in range(10):
        position[0, i * 10 + j] = (i * 0.5 + 12.75, j * 0.5 + 12.75)
    input_state = sim.get_initial_state(position=position)

    def center_of_mass():
      return np.mean(input_state[0][:, :, 0]), np.mean(input_state[0][:, :, 1])

    print(center_of_mass())
    for i in range(10):
      input_state = sess.run(next_state.to_tuples(), feed_dict={sim.initial_state_place_holder(): input_state})
      print(center_of_mass())

  def test_translation_batched(self):
    pass

if __name__ == '__main__':
  unittest.main()