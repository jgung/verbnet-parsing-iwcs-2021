import logging

import tensorflow as tf
from tensorflow.python.platform import tf_logging


def set_up_logging(log_path=None, level=tf.compat.v1.logging.INFO,
                   formatter='%(asctime)s - %(name)s - %(levelname)s - %(message)s'):
    tf_logging.set_verbosity(level)
    # create file handler which logs even debug messages
    if log_path:
        try:
            fh = logging.FileHandler(log_path)
            fh.setLevel(level)
            fh.setFormatter(logging.Formatter(formatter))

            tf_logger = tf_logging.get_logger()
            tf_logger.addHandler(fh)
            tf_logger.info('Saving logs to "%s"' % log_path)
            tf_logger.propagate = False
        except FileNotFoundError:
            tf.logging.info('Cannot save logs to file in Cloud ML Engine')
