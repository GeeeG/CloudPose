import tensorflow as tf
import sys

def get_translation_error(pred, label):

    loss_perSample = tf.sqrt(tf.reduce_sum(tf.square(label - pred), axis=1))
    loss = tf.reduce_mean(loss_perSample)

    return loss, loss_perSample