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

![MommyTalk1666611014838](https://user-images.githubusercontent.com/67272893/197516900-cd238e15-b471-4f33-ac63-1bbe5fb82586.jpg)

The loss of keypoint consists of two parts: the loss of $(x,y)$ and loss of $conf$.

The loss of $(x,y)$. Hence, if a ground truth bounding box is matched with $k_{th}$ anchor at location $(i,j)$ and scale $s$, we predict the keypoints with respect to the center of the anchor. 
![image](https://user-images.githubusercontent.com/67272893/197521672-c631778a-b07c-40f1-9286-6ac1907a9f11.png)

The loss of $conf$, visibility flags for keypoints are used as ground truth.
![image](https://user-images.githubusercontent.com/67272893/197521866-e7d16a62-2660-404e-8b65-25e5e742e333.png)

![image](https://user-images.githubusercontent.com/67272893/197498654-f5b66058-0f28-4339-8cee-e9eda2f27773.png)
