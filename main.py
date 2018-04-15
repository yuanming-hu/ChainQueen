import tensorflow as tf
import os
from simulation import Simulation
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def main(sess):
  t = time.time()
  sim = Simulation(sess)
  print("Building time: {:.4f}s".format(time.time() - t))
  t = time.time()
  sim.run()
  print("Running time: {:.4f}s".format(time.time() - t))


if __name__ == '__main__':
  sess_config = tf.ConfigProto(allow_soft_placement=True)
  sess_config.gpu_options.allow_growth = True
  sess_config.gpu_options.per_process_gpu_memory_fraction = 0.4

  with tf.Session(config=sess_config) as sess:
    main(sess=sess)
