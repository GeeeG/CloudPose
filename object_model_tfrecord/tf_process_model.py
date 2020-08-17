# examples/Python/Tutorial/Basic/pointcloud.py
# https://codereview.stackexchange.com/questions/179561/farthest-point-algorithm-in-python

import tensorflow as tf
import numpy as np
import open3d
import sys
import random

object_names = ["002_master_chef_can",
                "003_cracker_box",
                "004_sugar_box",
                "005_tomato_soup_can",
                "006_mustard_bottle",
                "007_tuna_fish_can",
                "008_pudding_box",
                "009_gelatin_box",
                "010_potted_meat_can",
                "011_banana",
                "019_pitcher_base",
                "021_bleach_cleanser",
                "024_bowl",
                "025_mug",
                "035_power_drill",
                "036_wood_block",
                "037_scissors",
                "040_large_marker",
                "051_large_clamp",
                "052_extra_large_clamp",
                "061_foam_brick"]


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def calc_distances(p0, points):
    return ((p0 - points)**2).sum(axis=1)


def FPS(pts, K):
    farthest_pts = np.zeros((K, 6))
    farthest_pts[0] = pts[0]
    distances = calc_distances(farthest_pts[0, 0:3], pts[:, 0:3])
    for i in range(1, K):
        farthest_pts[i] = pts[np.argmax(distances)]
        distances = np.minimum(distances, calc_distances(farthest_pts[i, 0:3], pts[:, 0:3]))
    return farthest_pts


def FPS_random(pts, K):
    farthest_pts = np.zeros((K, 6))
    farthest_pts_idx = np.zeros(K)
    first_idx = random.randint(0, pts.shape[0])
    farthest_pts[0] = pts[first_idx]
    farthest_pts_idx[0] = first_idx
    distances = calc_distances(farthest_pts[0, 0:3], pts[:, 0:3])
    for i in range(1, K):
        farthest_pts[i] = pts[np.argmax(distances)]
        farthest_pts_idx[i] = np.argmax(distances)
        distances = np.minimum(distances, calc_distances(farthest_pts[i, 0:3], pts[:, 0:3]))
    return farthest_pts, farthest_pts_idx


if __name__ == "__main__":

    K = 4096
    output_path = "/your/path/to/obj_models_dense.tfrecords"

    writer = tf.python_io.TFRecordWriter(output_path)

    for obj_name, cls in zip(object_names, range(0, len(object_names))):

        ply_path = "ycb_video_obj_ply/" + obj_name + ".ply"
        pcd = open3d.read_point_cloud(ply_path)
        print(pcd)

        xyz = np.asarray(pcd.points)
        rgb = np.asarray(pcd.colors)
        pts = np.concatenate((xyz, rgb), axis=1)
        # farthest_pts = FPS(pts, K)
        farthest_pts, farthest_pts_idx = FPS_random(pts, K)

        example = tf.train.Example(features=tf.train.Features(
            feature={
                'label': _int64_feature([cls]),
                'model': _float_feature(farthest_pts.reshape(-1))
            }
        ))

        writer.write(example.SerializeToString())

    writer.close()
