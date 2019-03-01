import tensorflow as tf
import numpy as np
import random
from scipy.io import loadmat
import transforms3d
import sys


def read_image(filename, dtype=tf.uint8):
    image_string = tf.read_file(filename)
    return tf.image.decode_png(image_string, dtype=dtype)


def read_data(x):
    x['image'] = read_image(x['color_file'])
    x['depth'] = read_image(x['depth_file'], dtype=tf.uint16)
    x['label'] = read_image(x['label_file'])
    return tf.data.Dataset.from_tensors(x)


def get_pointcloud(depth, fx, fy, cx, cy, depth_scaling_factor):
    depth_meters = tf.cast(depth, tf.float32) / depth_scaling_factor

    dshape = tf.shape(depth_meters)
    height = dshape[0]
    width = dshape[1]
    xv = tf.cast(tf.range(width), tf.float32)
    yv = tf.cast(tf.range(height), tf.float32)
    X, Y = tf.meshgrid(xv, yv)

    x = ((X - cx) * tf.squeeze(depth_meters) / fx)
    y = ((Y - cy) * tf.squeeze(depth_meters) / fy)
    xyz = tf.stack([x, y, tf.squeeze(depth_meters)], axis=2)  # (height, width, 3)

    return tf.reshape(xyz, [height * width, 3])


def merge_two_dicts(x, y):
    z = x.copy()  # start with x's keys and values
    z.update(y)  # modifies z with y's keys and values & returns None
    return z


def split_samples(x):
    xyz = get_pointcloud(x["depth"], x["fx"], x["fy"], x["cx"], x["cy"], x["factor_depth"])
    rgb = tf.reshape(tf.image.convert_image_dtype(x['image'], dtype=tf.float32), [-1, 3])
    hsv = tf.image.rgb_to_hsv(rgb)

    class_idx = tf.where(x["class_one_hot"])
    classes = tf.reshape(class_idx, [-1])

    depth_flat = tf.cast(tf.reshape(x["depth"], [-1]), tf.int64)
    depth_valid = tf.not_equal(depth_flat, 0)

    data_static = {'xyz': xyz,
                   'rgb': rgb,
                   'hsv': hsv,
                   'depth_valid': depth_valid,
                   'label': x["label"]
                   }
    d_static = tf.data.Dataset.from_tensors(data_static)

    data_dynamic = {'class_id': classes}
    d_dynamic = tf.data.Dataset.from_tensor_slices(data_dynamic)

    ds = tf.data.Dataset.zip((d_static, d_dynamic))
    ds = ds.map(lambda y, x: merge_two_dicts(y, x))
    return ds


def segment_mean_distance_filter(xyz, label_mask, threshold_distance):
    # Filtering based on distance from mean of segment
    segment_average_xyz = tf.reduce_mean(tf.boolean_mask(xyz, label_mask), axis=0)
    d = tf.norm(xyz-segment_average_xyz, ord='euclidean', axis=1)
    return tf.logical_and(label_mask, tf.less_equal(d, threshold_distance))


def calc_distances(p0, points):
    return ((p0 - points)**2).sum(axis=1)


def FPS_random(pts, K):
    farthest_pts = np.zeros((K, 3))
    farthest_pts_idx = np.zeros(K)
    upper_bound = pts.shape[0] - 1
    first_idx = random.randint(0, upper_bound)
    farthest_pts[0] = pts[first_idx]
    farthest_pts_idx[0] = first_idx
    distances = calc_distances(farthest_pts[0, 0:3], pts[:, 0:3])
    for i in range(1, K):
        farthest_pts[i] = pts[np.argmax(distances)]
        farthest_pts_idx[i] = np.argmax(distances)
        distances = np.minimum(distances, calc_distances(farthest_pts[i, 0:3], pts[:, 0:3]))
    return farthest_pts_idx.astype(np.int64)


# label exist for not presenting object
def segment_filter(x, threshold_distance_per_class):
    label_flat = tf.cast(tf.reshape(x["label"], [-1]), tf.int64) - 1  # To zero-based class indexing!
    label_mask = tf.logical_and(tf.equal(label_flat, x["class_id"]), x["depth_valid"])
    label_mask_r = segment_mean_distance_filter(x['xyz'], label_mask,
                                                threshold_distance=tf.gather(threshold_distance_per_class,
                                                                             x["class_id"]))
    x["label_mask_r"] = label_mask_r
    x["segment_after_filter"] = tf.count_nonzero(label_mask_r)
    return x


def segment_sample_FPS(x, numpoints, threshold_distance_per_class):
    label_mask_r = x["label_mask_r"]
    num_nonzeros = tf.count_nonzero(label_mask_r)
    idx = tf.py_func(FPS_random, [tf.boolean_mask(x['xyz'], label_mask_r), numpoints], tf.int64)

    y_out = {'class_id': x['class_id']}

    y_out["num_valid_points_in_segment"] = num_nonzeros
    y_out["xyz"] = tf.gather(tf.boolean_mask(x['xyz'], label_mask_r), idx)
    y_out["rgb"] = tf.gather(tf.boolean_mask(x['rgb'], label_mask_r), idx)
    y_out["hsv"] = tf.gather(tf.boolean_mask(x['hsv'], label_mask_r), idx)

    return y_out
