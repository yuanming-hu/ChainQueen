import unittest
from simulation import Simulation
from time_integration import SimulationState, InitialSimulationState, UpdatedSimulationState
import tensorflow as tf

sess = tf.Session()

class TestSimulator(unittest.TestCase):

  def test_acceleration(self):
    pass

  def test_free_fall(self):
    pass

  def test_translation(self):
    # Zero gravity, 1-batched, translating block
    sim = Simulation(grid_res=(30, 30), num_particles=100)
    initial = sim.initial_state
    next_state = UpdatedSimulationState(sim, initial)
    initial_inputs =
    sess.eval(next_state, initial)

  def test_translation_batched(self):
    pass

if __name__ == '__main__':
  unittest.main()