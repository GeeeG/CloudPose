import tensorflow as tf
import math
import sys


def skew_symmetric(axag_unit):
    """
    Create the skew symmetric matrix for the input vector
    v = (v_1, v_2, v_3)
    v_ss = | 0    -v_3    v_2 |
           | v_3     0   -v_1 |
           | -v_2  v_1     0  |

    :param axag_unit: B, 3 tensor
    :return: B, 3, 3 tensor
    """

    sh = axag_unit.shape
    axag_unit_exp = tf.expand_dims(tf.expand_dims(axag_unit, 2), 3)

    row1 = tf.concat([tf.zeros((sh[0], 1, 1), dtype=tf.float64), -axag_unit_exp[:, 2, :, :], axag_unit_exp[:, 1, :, :]], axis=2)
    row2 = tf.concat([axag_unit_exp[:, 2, :, :], tf.zeros((sh[0], 1, 1), dtype=tf.float64), -axag_unit_exp[:, 0, :, :]], axis=2)
    row3 = tf.concat([-axag_unit_exp[:, 1, :, :], axag_unit_exp[:, 0, :, :], tf.zeros((sh[0], 1, 1), dtype=tf.float64)], axis=2)

    axag_unit_ss = tf.concat([row1, row2, row3], axis=1)

    return axag_unit_ss


def exponential_map(axag, EPS=1e-2):
    """
    Create exponential map for axis-angle representation using Rodrigues' formula
    axag = theta * v_hat
    exp(theta * v_hat) = I + sin(theta)[v_hat]_x + (1 - cos(theta))([v_hat]_x)^2
    For small angle values, use Taylor expansion
    :param axag: B, 3 tensor
    :return: B, 3, 3 tensor
    """
    ss = skew_symmetric(axag)

    theta_sq = tf.reduce_sum(tf.square(axag), axis=1)

    is_angle_small = tf.less(theta_sq, EPS)

    theta = tf.sqrt(theta_sq)
    theta_pow_4 = theta_sq * theta_sq
    theta_pow_6 = theta_sq * theta_sq * theta_sq
    theta_pow_8 = theta_sq * theta_sq * theta_sq * theta_sq

    term_1 = tf.where(is_angle_small,
                      1 - (theta_sq / 6) + (theta_pow_4 / 120) - (theta_pow_6 / 5040) + (theta_pow_8 / 362880),
                      tf.sin(theta) / theta)

    term_2 = tf.where(is_angle_small,
                      0.5 - (theta_sq / 24) + (theta_pow_4 / 720) - (theta_pow_6 / 40320) + (theta_pow_8 / 3628800),
                      (1 - tf.cos(theta)) / theta_sq)

    term_1_expand = tf.expand_dims(tf.expand_dims(term_1, 1), 2)
    term_2_expand = tf.expand_dims(tf.expand_dims(term_2, 1), 2)
    batch_identity = tf.eye(3, batch_shape=[axag.shape[0]], dtype=tf.float64)

    axag_exp = batch_identity + tf.multiply(term_1_expand, ss) + tf.multiply(term_2_expand, tf.matmul(ss, ss))

    # print axag_exp.shape

    return axag_exp


def logarithm(R, b_deal_with_sym=False, EPS=1e-2):
    """
    R in SO(3)
    theta = arccos((tr(R)-1)/2)
    ln(R) = (theta/(2*sin(theta)))*(R-R.')
    :param R: B, 3, 3 tensor
    :return: B, 3 tensor
    """
    trace = tf.trace(R)
    trace_temp = (trace - 1) / 2

    # take the safe acos
    trace_temp = tf.clip_by_value(trace_temp, -0.9999999, 0.9999999)

    theta = tf.acos(trace_temp)

    is_angle_small = tf.less(theta, EPS)
    theta_pow_2 = theta * theta
    theta_pow_4 = theta_pow_2 * theta_pow_2
    theta_pow_6 = theta_pow_2 * theta_pow_4

    ss = (R - tf.matrix_transpose(R))

    mul_expand = tf.where(is_angle_small,
                          0.5 + (theta_pow_2 / 12) + (7 * theta_pow_4 / 720) + (31 * theta_pow_6 / 30240),
                          theta / (2 * tf.sin(theta)))
    if b_deal_with_sym:
        log_R = tf.expand_dims(tf.expand_dims(mul_expand, 2), 3) * ss
    else:
        log_R = tf.expand_dims(tf.expand_dims(mul_expand, 1), 2) * ss

    return log_R, theta


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

