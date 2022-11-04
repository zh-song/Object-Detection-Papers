

## Dataloader and lables





## Detect Head framework 

```
class Detect(nn.Module):
	def __init__(self, nc=80, anchors=(), nkpt=None, ch=(), inplace=True, dw_conv_kpt=False):
        ...
        self.nkpt = nkpt # num of keypoint
        self.dw_conv_kpt = dw_conv_kpt # subbranch of predict keypoint
        self.no_kpt = 3*self.nkpt ## number of outputs per anchor for keypoints
        ...
        if self.nkpt is not None: # predict keypoint [x,y,conf] module with different architecture
            if self.dw_conv_kpt: #keypoint head is slightly more complex
                self.m_kpt = nn.ModuleList(
                            nn.Sequential(DWConv(x, x, k=3), Conv(x,x),
                                          DWConv(x, x, k=3), Conv(x, x),
                                          DWConv(x, x, k=3), Conv(x,x),
                                          DWConv(x, x, k=3), Conv(x, x),
                                          DWConv(x, x, k=3), Conv(x, x),
                                          DWConv(x, x, k=3), nn.Conv2d(x, self.no_kpt * self.na, 1)) for x in ch)
            else: #keypoint head is a single convolution
                self.m_kpt = nn.ModuleList(nn.Conv2d(x, self.no_kpt * self.na, 1) for x in ch)
                
    def forward(self, x):
    	...
    	for i in range(self.nl):
            if self.nkpt is None or self.nkpt==0:
                x[i] = self.m[i](x[i])
            else :
                x[i] = torch.cat((self.m[i](x[i]), self.m_kpt[i](x[i])), axis=1)
                
            # x (bs,num_anchor,feature_h,feature_w,5+nc)
            
            # only detect object and keypoint for person this class which num_class is 1 
            x_det = x[i][..., :6] # x,y,w,h,conf,cls
            x_kpt = x[i][..., 6:] # x_k,y_k,conf_k
            
            ...
            
            	y = x_det.sigmoid()
            	
            	# decode keypoint x and y with grid and stride, and conf of this keypoint
                x_kpt[..., 0::3] = (x_kpt[..., ::3] * 2. - 0.5 + kpt_grid_x.repeat(1,1,1,1,17)) * self.stride[i]  # xy
                x_kpt[..., 1::3] = (x_kpt[..., 1::3] * 2. - 0.5 + kpt_grid_y.repeat(1,1,1,1,17)) * self.stride[i]  # xy
                x_kpt[..., 2::3] = x_kpt[..., 2::3].sigmoid()
                
                y = torch.cat((xy, wh, y[..., 4:], x_kpt), dim = -1)
```



## Loss

```
class ComputeLoss:
    # Compute losses
    def __init__(self, model, autobalance=False, kpt_label=False):
        super(ComputeLoss, self).__init__()
        self.kpt_label = kpt_label
		...

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))
        BCE_kptv = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))
```



## NMS

```
                ...
                kpts = x[:, 6:]
                conf, j = x[:, 5:6].max(1, keepdim=True)
                x = torch.cat((box, conf, j.float(), kpts), 1)[conf.view(-1) > conf_thres]
                ...
```

