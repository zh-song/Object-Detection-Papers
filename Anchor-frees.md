## **CornerNet**

> corner pooling（以左上角为例）
 
    两个不同特征图上不同方向进行maxpooling
    大小：C×H×W，C表示物体类别；HW上每个位置表示每个点是否为角点的概率

![image](https://user-images.githubusercontent.com/67272893/147025619-c917e74e-e037-4eb5-be36-364eadf73e0f.png)

> 预测输出

    heatmap：成为角点的概率
    embedding：计算两角点embedding之间的距离，判断是否为一组角点
    offset：修正的偏移量

![image](https://user-images.githubusercontent.com/67272893/147028761-5d39888a-0b33-4263-84da-59125109ef4d.png)

> 网络结构

    Hourglass Networks：先对图像下采样再上采样恢复尺寸大小获得全局信息，同时使用shortcut补全细节信息
    
![image](https://user-images.githubusercontent.com/67272893/147030965-e4e376d3-1134-44ce-a320-0b7f86ba7d7e.png)


## **CenterNet**

>  Key words: a triplet of keypoints; cascade corner pooling and center pooling

      Center pooling:
      Cascade corner pooling:从角点沿着边界找最大值，
      
![image](https://user-images.githubusercontent.com/67272893/147032059-e2a251f2-5645-4d2c-ac31-2661d5370df1.png)
