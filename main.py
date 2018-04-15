import tensorflow as tf
import os
from simulation import Simulation
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main(sess):
  sim = Simulation(sess)
  sim.run()


if __name__ == '__main__':
  sess_config = tf.ConfigProto(allow_soft_placement=True)
  sess_config.gpu_options.allow_growth = True
  sess_config.gpu_options.per_process_gpu_memory_fraction = 0.4

  with tf.Session(config=sess_config) as sess:
    main(sess=sess)
