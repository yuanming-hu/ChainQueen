import tensorflow as tf
import cv2
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

batch_size = 1
particle_count = 100
gravity = (0, -9.8)


def polar_decomposition(m):
  assert False

class States:
  def __init__(self):
    self.position = tf.placeholder(tf.float32, [batch_size, particle_count, 2], name='position')
    self.velocity = tf.placeholder(tf.float32, [batch_size, particle_count, 2], name='velocity')
    self.deformation_gradient = tf.placeholder(tf.float32, [batch_size, particle_count, 4], name='dg')

    '''
    TODO:
    mass, volume, Lame parameters (Young's modulus and Poisson's ratio)
    '''

  


def main():
  sess_config = tf.ConfigProto(allow_soft_placement=True)
  sess_config.gpu_options.allow_growth = True
  sess_config.gpu_options.per_process_gpu_memory_fraction = 0.4

  sess = tf.Session(config=sess_config)

  x = tf.placeholder(tf.float32, [batch_size, 1], name='x')
  y0 = tf.placeholder(tf.float32, [batch_size, 1], name='y0')


if __name__ == '__main__':
  main()