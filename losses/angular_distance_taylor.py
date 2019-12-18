import tensorflow as tf
import math
import sys


def get_rotation_error(pred, label):
    '''
    Return (mean) rotation error in form of angular distance in SO(3)
    :param pred: B,3 tensor
    :param label: B,3 tensor
    :return: 1D scalar
    '''
    pred_expMap = exponential_map(pred)
    label_expMap = exponential_map(label)

    R = tf.matmul(label_expMap, tf.matrix_transpose(pred_expMap))
    R_logMap, loss = logarithm(R)

    return tf.reduce_mean(loss), loss
