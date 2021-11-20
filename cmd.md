> 指定GPU运行程序
    
    CUDA_VISIBLE_DEVICES=0 nohup python detect.py >nohup1.out 2>&1 &

> 切换端口

     ssh 192.168.8.206
     
> 查看GPU使用情况

    nvidia-smi
