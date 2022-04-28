# DETR

![Fig2](https://pdf.cdn.readpaper.com/parsed/fetch_target/fe95d3164d4c868cd30406f698068bdf_6_Figure_2.png)

<img src="/home/zhsong/snap/typora/57/.config/Typora/typora-user-images/image-20220420164005554.png" alt="image-20220420164005554" style="zoom:80%;" />

<img src="/home/zhsong/snap/typora/57/.config/Typora/typora-user-images/image-20220420163939780.png" alt="image-20220420163939780" style="zoom:50%;" />

DETR 的 cross-attention 有三个输入：**query, key, value**。

1. Query 由来自 decoder 中 self-attention 的输出 (content query: ![[公式]](https://www.zhihu.com/equation?tex=c_q)) 和所有图片共享的 object query (spatial query: ![[公式]](https://www.zhihu.com/equation?tex=p_q), 在DETR中其实就是 object query ![[公式]](https://www.zhihu.com/equation?tex=o_q)) 相加得到。
2. Key 由来自 encoder 的输出 (content key: ![[公式]](https://www.zhihu.com/equation?tex=c_k)) 和对于 2D 坐标的位置编码 (spatial key: ![[公式]](https://www.zhihu.com/equation?tex=p_k)) 相加得到。Value 的组成和 key 相同。

在这里，**`content`** 代表这个向量的内容和图像 (颜色、纹理等) 是相关的，而 **`spatial`** 代表这个向量它更多包含空间上的信息，他的内容和图像的内容无关。Attention 模块的输出，就是对 query 和 key  算一次内积得到注意力的权重，用这个权重给 value 进行加权。我们将这个过程写成下面的形式：

![[公式]](https://www.zhihu.com/equation?tex=%28c_q%2Bp_q%29%5E%5Ctext%7BT%7D%28c_k%2Bp_k%29+%5C%5C+%3D+c_q%5E%5Ctext%7BT%7Dc_k+%2B+c_q%5E%5Ctext%7BT%7Dp_k+%2B+p_q%5E%5Ctext%7BT%7Dc_k+%2B+p_q%5E%5Ctext%7BT%7Dp_k+%5C%5C+%3D+c_q%5E%5Ctext%7BT%7Dc_k+%2B+c_q%5E%5Ctext%7BT%7Dp_k+%2B+o_q%5E%5Ctext%7BT%7Dc_k+%2B+o_q%5E%5Ctext%7BT%7Dp_k+%5Ctag%7B1%7D+)

Head of query-based has 2 steps:

1. > **Label assignment** 

   ***DETR:***

   <img src="https://img-blog.csdnimg.cn/20200611091538612.png" alt="bipartite_matching" style="zoom:80%;" />

   <img src="https://img-blog.csdnimg.cn/img_convert/9586fafaa8ccbd9aec06ca8f4c1ed4ee.png" alt="match loss" style="zoom:80%;" />

2. >  **Compute loss**

   ***DETR:***

   <img src="https://img-blog.csdnimg.cn/20200611091601304.png" alt="Hungarian loss" style="zoom:80%;" />

   <img src="https://img-blog.csdnimg.cn/20200611091615382.png" alt="box loss" style="zoom:80%;" />

# Deformable DETR

将原来的**全局注意力**计算改为**参考点($p$)周围的局部($\Delta p_{mqk}$)注意力计算**，并采用多尺度特征 。

以qkv角度来说：query不是和全局每个位置的key都计算注意力权重，而是**对于每个query，仅在全局位置中采样部分位置的key，并且value也是基于这些位置进行采样插值得到的**，最后将这个**局部&稀疏**的注意力权重施加在对应的value上。

> **公式差异**

<img src="https://img-blog.csdnimg.cn/20210713104713513.png" alt="在这里插入图片描述" style="zoom:50%;" />

<img src="https://img-blog.csdnimg.cn/2021071311205178.png" alt="在这里插入图片描述" style="zoom: 67%;" />

**注**：![[公式]](https://www.zhihu.com/equation?tex=x%28p_%7Bq%7D%2B%5CDelta+p_%7Bmqk%7D%29) 代表基于采样点位置插值出来的value

> **实现方式**

将参考点映射到不同尺度特征图，同时给予scale-level embedding

在多个level上采样，每个level采k个，共采样$l\times k$

![img](https://img2020.cnblogs.com/blog/2541889/202109/2541889-20210921142314350-1599812963.png)

> Encoder

**Transformer encoder input**: feature map + position embedding + scale-level embedding

1. 参考点($p_q$)：每个特征图中点的坐标经过归一化 or **多尺度特征点的归一化坐标**，由objdect query直接通过linear预测输出
2. 偏移量($\Delta p_{mqk}$)：query经过linear输出,初始的采样点位置相当于会分布在参考点3x3、5x5、7x7、9x9方形邻域
3. **$A_{mqk}$**：不经过QK相乘，直接linear输出
4. Query($z_p$)：object query(特征图点) + query embedding(position embedding + scale-level embedding)
5. Key or Value：$l\times k$ 个采样点+位置编码
6. Attention全是MS-Self-Attention（MS代表多尺寸）,

> Decoder

**将参考点坐标映射（re-scales）到各尺度特征层**

**Self-attention** （学习各个object目标之间的关系）

1. *query or key*: object query+query embedding
2. *value*: object query（注意不需要位置嵌入哦）

**Cross-attention** （object queries从encoder输出的feature map中提取特征（key，value））

只将decoder cross-attention模块替换为多尺度可变形注意力模块，而decoder self-attention模块仍沿用Transformer本身的。

1. *query*: object query+query embedding

2. *key:* encoder输出的多尺度特征图像素点（输入query的feature直接回归LK个偏差，LK个贡献图）

3. *value*: 由Encoder编码的feature经过线性变换得到（在query基础上根据偏差聚合特征，作为value）

   每一个value都通过多尺度deformable attention module汇聚了特征，设计检测头预测的是以key为中心的bbox偏置

<img src="https://img-blog.csdnimg.cn/20210713103936679.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3hpanVlemh1ODEyOA==,size_16,color_FFFFFF,t_70" alt="img" style="zoom:60%;" />

Reference:

1.  [code and note](https://zhuanlan.zhihu.com/p/372116181) 
2. [paper notes](https://blog.csdn.net/xijuezhu8128/article/details/118693939#t8)

# Conditional DETR

| <img src="/home/zhsong/snap/typora/57/.config/Typora/typora-user-images/image-20220420164005554.png" alt="image-20220420164005554" style="zoom:79%;" /> | <img src="/home/zhsong/snap/typora/57/.config/Typora/typora-user-images/image-20220420194401715.png" alt="image-20220420194401715" style="zoom: 75%;" /> |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
|                             DETR                             |                       Conditional DETR                       |

主要区别在于cross-attention

1. 将`content query` $\oplus$ `spatial query`变为concat
2. 将`content`(关注目标图像内容)和`spatial`(关注目标空间位置)解耦

1. 从decoder embedding中提取与reference point的偏移量 $T$，类似于anchor框与中心点偏移，具体来说通过$FFN(decoder\ embedding)$来实现；
2. 再将从object query中预测得出的$s$（2维坐标点类似与anchor中心点）映射到与saptial positional encoding相同的正弦位置编码空间中得 $p_s$，将 $p_s$ 和 $T$ 相乘得到spatial query。

Reference：

1. [paper note with author](https://zhuanlan.zhihu.com/p/401916664)

# Efficient DETR

在 backbone 出来的 dense feature map (`shape: [256, h, w]`)上预测 top-k score 的 proposals, 用这些top-k score(`shape: [2, h, w]`)的idx来对索引 proposal 的 2-d 或者 4-d 坐标用于初始化 reference point. 用同样的idx索引选择 top-k 的 feature (`shape: [256, h, w]`)来初始化 object  query.

Reference points are 2-d tensors that represent predictions of box centers and belong to the location information of an object container. 

不同Reference points初始化方式，对于1-decoder最终效果影响差异巨大，但对cascade-decoder影响不大，所以采用1-decoder和特殊的初始化方式来减小和cascade-decoder之间的gap。

![image-20220421203918815](/home/zhsong/snap/typora/57/.config/Typora/typora-user-images/image-20220421203918815.png)

# Anchor DETR

1. 改变位置编码，对于同一区域多个目标，同一个点采用多个模式($N_p$)编码，采用$sin$编码，$$Q^i_f\in R^{N_A\times 1\times C}\to Q^i_f\in R^{N_A\times N_q\times C}$$

<img src="/home/zhsong/Pictures/Screenshot from 2022-04-22 15-49-34.png" alt="Screenshot from 2022-04-22 15-49-34" style="zoom:67%;" />

<img src="/home/zhsong/Pictures/Screenshot from 2022-04-22 15-52-59.png" style="zoom:67%;" />

<img src="/home/zhsong/Pictures/Screenshot from 2022-04-22 15-49-50.png" alt="Screenshot from 2022-04-22 15-49-50" style="zoom:67%;" />

<img src="/home/zhsong/Pictures/Screenshot from 2022-04-22 15-49-59.png" style="zoom:67%;" />

2. Row-Column Decoupled Attention；将2D注意力解耦为x和y两个1D注意力，减少内存消耗(即用一维的gobal pooling消除纵向或者横向的维度)

<img src="/home/zhsong/Pictures/Screenshot from 2022-04-22 15-47-26.png" style="zoom: 67%;" />

# Dynamic DETR

encoder：用中间层的deformable 偏移作用到所有level feature map上，用SE layer通道注意力加权一波，relu再cat

ROI：ROI window内局部信息，缺少全局特征

![image-20220428204600257](/home/zhsong/snap/typora/57/.config/Typora/typora-user-images/image-20220428204600257.png)

# Sparse R-CNN

<img src="/home/zhsong/snap/typora/57/.config/Typora/typora-user-images/image-20220428205107332.png" alt="image-20220428205107332" style="zoom:67%;" />

1. N个anchor类似于anchor DETR中多模式
2. 动态refine box，head本质作attention

<img src="/home/zhsong/snap/typora/57/.config/Typora/typora-user-images/image-20220428205248401.png" alt="image-20220428205248401" style="zoom:67%;" />

[paper note]: https://zhuanlan.zhihu.com/p/409121670



