# CloudPose
This is an tensorflow implementation of "Learning Object Pose Estimation with Point Clouds"
![](figure/System_fig.png?raw=true)
# Files
1. **data**: a color, a depth and a label image for testing.
2. **models**: a python file defining model structure.
3. **object_model_tfrecord**: full object models for visualization purpose.
4. **trained_network**: a trained network.
5. **utils**: utility files for defining model structure.
6. **log**: directory to store log files during training.
7. **losses**: loss functions for training.
8. *data_process_tools.py*: tools for data pre-processing.
9. *object_6d_pose.py*: main file for testing object 6d pose estimation with a trained network.
10. *train_6d_pose.py*: script for training a network.


# Requirements
* Tensorflow-GPU (tested with 1.12.0)
* [transforms3d](https://matthew-brett.github.io/transforms3d/)
* [open3d](http://www.open3d.org/docs/getting_started.html) for visualization

# Test a trained network
1. Available classes in the test scenario are: 
 * 1: 003_cracker_box
 * 3: 005_tomato_soup_can
 * 4: 006_mustard_bottle
 * 9: 011_banana
 * 14: 035_power_drill
2. After activate tensorflow
```
python object_6d_pose.py --trained_model trained_network/20190222-130143/model.ckpt --batch_size 1 --target_class 9
```
* --trained_model: directory to trained model (*.ckpt).
* --batch_size: set to 1 because only 1 test example is provided in this test.
* --target_class: target class for pose estimation.
* Translation prediction is in unit meter.
* Rotation prediction is in axis-angle format.
3. Result
* If you turn on visualization with **b_visual=True**, you will see the following displays which are partially observed point cloud segments (colored) overlaid with object model (green) with pose estimates. 
* The coordinate is the **object coordinate**, object segment is viewed in the **camera coordinate** 
<p float="center">
  <img src="/figure/1.png" width="150" />
  <img src="/figure/3.png" width="150" /> 
  <img src="/figure/4.png" width="150" />
  <img src="/figure/9.png" width="150" />
  <img src="/figure/14.png" width="150" />
</p>

# Train a network
1. Training data in **tfrecord** format is available
* Download [zip file](https://drive.google.com/file/d/10gvvJdllvl0mMGvpiwbZ8KEzVlJS2gyZ/view?usp=sharing)
* Unzip and place all 42 files under **ycb_video_tfRecords/FPS1024/**
2. Run script
```
python train_6d_pose.py
```
# Citation
Further details are available in our paper on the subject. If you use this code in an academic context, please consider cite the paper:
[arxiv version](https://arxiv.org/abs/2001.08942)

# Video
[![Video](https://img.youtube.com/vi/OhNMngzxFDQ/0.jpg)](https://www.youtube.com/watch?v=OhNMngzxFDQ)


BiBTeX:
```
@inproceedings{gao2020cloudpose,
      title={6D Object Pose Regression via Supervised Learning on Point Clouds},
      author={G. Gao, M. Lauri, Y. Wang, X. Hu, J. Zhang and S. Frintrop},
      booktitle={ICRA},
      year={2020}
    }
```

# Acknowledgement
* The building block for this system is [PointNet](https://github.com/charlesq34/pointnet).
