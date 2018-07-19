# SORT
This repo is cloned from https://github.com/abewley/sort initially.
I modified it a little for research.

### Environment:
- Hardware
    - GPU needed
- Software
    - Ubuntu 16.04
    - Python2.7

### Dependencies:

1. Numpy1.14.1
2. Matplotlib
3. Opencv3.4.1 (compiled with ffmpeg support, otherwise you cannot read/write videos) <- important!
and other needed libraries (run the scripts and you'll know what to install)
If you are running on the server under my account, use my virtual env directly
Example:
```sh
$ conda env list
$ source activate py27
$ python sort.py
```
### Steps of usage:

1. Prepare your video data
2. Detect and store the detection results in MOT format
3. Use the detection results to track

##### Video data example:
path/to/video/
- 000001.jpg
- 000002.jpg
- 000003.jpg
- ...
##### How to detect by YOLO?
```sh
$ cd ..
$ mkdir det_mot
$ python detect_sort.py
```
Detection results will be store in det_mot/

##### Track:
Modify the image and detection paths in sort.py. (line275~line280)
```sh
$ mkdir kf_output
$ python sort.py
```

### File description:

##### Utility functions
- bbox_utils.py
- utils.py

##### SORT scripts
- sort.py: basically original sort.py
- reid_sort.py: SORT(ReID)
    - Need scncd
- (sort_ukf.py: a little try of Unscented Kalman filter)
- reidsort_velicity.py: use this file for predicting target's velocity and position

##### Visualization
- output2video.py:
    - convert mot_format .txt results to video
- output2video_velocity.py
    - convert mot_format .txt results to video with velocity prediction

##### Evaluation
- (evaluation.py: contains Average IoU, Area Under Curve)
    - (I seldom used the script before graduation.)

##### Darknet
**libdarknet.a** and **libdarknet.so** are needed for using YOLO

##### Online tracking (Detect and track in one steps)
- online_tracking.py
- online_tracking_trajectory.py

