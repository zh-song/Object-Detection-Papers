# Top-down
1. Employ a heavy person detector
2. Single person pose estimation for each detection

# Bottom-up
1. Predict the heatmaps to detect all the keypoint
2. At the same time, group all the keypoint into individual persons
3. Postprocessing: pixel-level NMS, line integral, refinement, grouping.


## YOLO-Pose
$P_{v}=\left\{C_{x}, C_{y}, W, H, \text { box }_{\text {conf }}, \text { class }_{\text {conf }}, K_{x}^{1}, K_{y}^{1}, K_{\text {conf }}^{1}, \ldots\right.K_{x}^{n}, K_{y, l 1}^{n}, K_{c a n f}^{n} }$


![image](https://user-images.githubusercontent.com/67272893/197498654-f5b66058-0f28-4339-8cee-e9eda2f27773.png)
