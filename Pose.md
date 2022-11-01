# Top-down
1. Employ a heavy person detector
2. Single person pose estimation for each detection

# Bottom-up
1. Predict the heatmaps to detect all the keypoint
2. At the same time, group all the keypoint into individual persons
3. Postprocessing: pixel-level NMS, line integral, refinement, grouping.

# Multi-task
## YOLO-Pose
For each anchor, the box head predicts the $(x,y,w,h,class score)$, the keypoint head predicts the 17 keypoints of person.

Each keypoint includes $(x,y,conf)$. 

Ground truth of keypoint conf: If a keypoint is either visible or occluded, then the ground truth confidence is set to 1 else if it is outside the field of view, confidence is set to zero.

<p align="center">
  <img width="460" height="300" src="https://user-images.githubusercontent.com/67272893/197978288-02aeac17-d291-422a-8fc5-e031cd1b63ba.png">
</p>

The loss of keypoint consists of two parts: the loss of $(x,y)$ and loss of $conf$.

The loss of $(x,y)$. Hence, if a ground truth bounding box is matched with $k_{th}$ anchor at location $(i,j)$ and scale $s$, we predict the keypoints with respect to the center of the anchor. 

The loss of $conf$, visibility flags for keypoints are used as ground truth.

<p align="center">
  <img src="https://user-images.githubusercontent.com/67272893/197978359-e8c73af9-95e4-4f9d-b710-2ef449e7a036.png">
</p>

![Screenshot from 2022-10-26 16-42-59](https://user-images.githubusercontent.com/67272893/197978695-cceee35a-c65a-4c16-9379-c6ef826513dd.png)


## DirectPose: Direct End-to-End Multi-Person Pose Estimation
The regression-based method have the potential to detect very dense keypoints.

The heatmap-based task is only used as an auxiliary loss during training.
![Screenshot from 2022-10-26 16-39-38](https://user-images.githubusercontent.com/67272893/197977999-b29e71e6-800c-49c6-98e3-9943052be71b.png)
**KPAlign**

Locator predict the rough locations of the keypoints from high-level features with a larger receptive field.

Sampler samples feature acorrding to the above offsets from high-resolution low-level features with a smaller receptive field.

Predictor make the final keypoint predictions.
![Screenshot from 2022-10-26 16-57-45](https://user-images.githubusercontent.com/67272893/197982412-af2fc4c1-495b-47c3-9534-1fc0228f52d0.png)
**Heatmap Prediction**

Previous heatmap-based keypoint detection methods [1] generate unnormalized Gaussian distribution centered at each keypoint. we perform a per-pixel classification here for simplicity. Note that we make use of multiple binary classifiers (i.e., one-versus-all) and therefore the number of output channels is K instead of K + 1.

**GT of heatmap**

On the heatmaps, if a location is the nearest location to a keypoint with type t, the classification label for the location is set as t, 

where t ∈ {1, 2, ..., K}. Otherwise, the label is 0.

## Single-Stage Multi-Person Pose Machines
**Structured pose representation**
1. predict displacements between body joints and the root joint.
2. we exploit the person centroid as the root joint of the person instance.

**Hierarchical SPR**

we divide the root joint and body joints into four hierarchies based on articulated kinematics [20] by their degrees of freedom and extent of deformation.

**GT of root joint**: 高斯heatmap

**GT of displacements**: Root Joint 為中心, τ為半徑範圍內的點到 body joints 的位移向量
