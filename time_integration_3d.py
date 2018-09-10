import numpy as np
import tensorflow as tf
import mpm3d

dim = 3
kernel_size = 3


class SimulationState3D:

  def __init__(self, sim):
    self.sim = sim
    self.affine = tf.zeros(shape=(self.sim.batch_size, dim, dim, sim.num_particles))
    self.position = None
    self.particle_mass = None
    self.particle_volume = None
    self.youngs_modulus = None
    self.poissons_ratio = None
    self.velocity = None
    self.deformation_gradient = None
    self.controller_states = None
    self.grid_mass = None
    self.grid_velocity = None
    self.step_count = None
    self.kernels = None
    self.debug = None

  def center_of_mass(self, left = None, right = None):
    return tf.reduce_sum(self.position[:, :, left:right] * self.particle_mass[:, :, left:right], axis=2) *\
           (1 / tf.reduce_sum(self.particle_mass[:, :, left:right], axis=2))

  def get_state_names(self):
    return [
        'position', 'velocity', 'deformation_gradient', 'affine',
        'particle_mass', 'particle_volume', 'youngs_modulus', 'poissons_ratio', 'step_count'
    ]

  def get_evaluated(self):
    # # batch, particle, dimension
    # assert len(self.position.shape) == 3
    # assert len(self.position.shape) == 3
    # # batch, particle, matrix dimension1, matrix dimension2
    # assert len(self.deformation_gradient.shape) == 4
    # # batch, x, y, dimension
    # assert len(self.grid_mass.shape) == 4
    # assert len(self.grid_velocity.shape) == 4

    ret = {
        'affine': self.affine,
        'position': self.position,
        'velocity': self.velocity,
        'deformation_gradient': self.deformation_gradient,
        'controller_states': self.controller_states,
        'grid_mass': self.grid_mass,
        'grid_velocity': self.grid_velocity,
        'kernels': self.kernels,
        'particle_mass': self.particle_mass,
        'particle_volume': self.particle_volume,
        'youngs_modulus': self.youngs_modulus,
        'poissons_ratio': self.poissons_ratio,
        'step_count': self.step_count,
        'debug': self.debug
    }
    ret_filtered = {}
    for k, v in ret.items():
      if v is not None:
        ret_filtered[k] = v
    return ret_filtered

  def to_tuple(self):
    evaluated = self.get_evaluated()
    return tuple([evaluated[k] for k in self.get_state_names()])

  def __getitem__(self, item):
    return self.get_evaluated()[item]


class InitialSimulationState3D(SimulationState3D):

  def __init__(self, sim, controller=None):
    super().__init__(sim)
    self.t = 0
    num_particles = sim.num_particles
    self.position = tf.placeholder(
        tf.float32, [self.sim.batch_size, dim, num_particles], name='position')

    self.velocity = tf.placeholder(
        tf.float32, [self.sim.batch_size, dim, num_particles], name='velocity')
    self.deformation_gradient = tf.placeholder(
        tf.float32, [self.sim.batch_size, dim, dim, num_particles], name='dg')
    self.particle_mass = tf.placeholder(
        tf.float32, [self.sim.batch_size, 1, num_particles],
        name='particle_mass')
    self.particle_volume = tf.placeholder(
        tf.float32, [self.sim.batch_size, 1, num_particles],
        name='particle_volume')
    self.youngs_modulus = tf.placeholder(
        tf.float32, [self.sim.batch_size, 1, num_particles],
        name='youngs_modulus')
    self.poissons_ratio = tf.placeholder(
        tf.float32, [self.sim.batch_size, 1, num_particles],
        name='poissons_ratio')

    self.grid_mass = tf.zeros(shape=(self.sim.batch_size, self.sim.grid_res[0],
                                     self.sim.grid_res[1], self.sim.grid_res[2], 1))
    self.grid_velocity = tf.zeros(
        shape=(self.sim.batch_size, self.sim.grid_res[0], self.sim.grid_res[1], self.sim.grid_res[2],
               dim))
    self.step_count = tf.zeros(shape=(), dtype=np.int32)

    self.controller = controller
    if controller is not None:
      self.actuation, self.debug = controller(self)


class UpdatedSimulationState3D(SimulationState3D):
  def __init__(self, sim, previous_state, controller=None):
    super().__init__(sim)
    self.particle_mass = tf.identity(previous_state.particle_mass)
    self.particle_volume = tf.identity(previous_state.particle_volume)
    self.youngs_modulus = tf.identity(previous_state.youngs_modulus)
    self.poissons_ratio = tf.identity(previous_state.poissons_ratio)

    self.step_count = previous_state.step_count + 1

    self.t = previous_state.t + self.sim.dt

    self.grid_velocity = tf.zeros(
        shape=(self.sim.batch_size, self.sim.grid_res[0], self.sim.grid_res[1],
               dim))
    
    self.position, self.velocity, self.deformation_gradient, self.affine, _, _ =\
      mpm3d.mpm(previous_state.position, previous_state.velocity,
                previous_state.deformation_gradient, previous_state.affine)
