import numpy as np
import tensorflow as tf

def xavier_init(fan_in, fan_out, constant=1):
    low = -constant*np.sqrt(6.0/(fan_in + fan_out))
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval=low, maxval=high,
                             dtype=tf.float32)

def log_dir_init(fan_in, fan_out, topics=50):
    return tf.log((1.0/topics)*tf.ones([fan_in, fan_out]))