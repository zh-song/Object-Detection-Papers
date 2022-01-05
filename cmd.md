> 指定GPU运行程序
    
    CUDA_VISIBLE_DEVICES=0 nohup python detect.py >cmdout.out 2>&1 &

> 切换端口

     ssh 192.168.8.206
     
> 查看GPU使用情况

    nvidia-smi

> 刷新GPU使用

    watch --color gpustat --color

> ptvsd

    python -m ptvsd --host 222.131.60.5 --port 205 --wait test_ptvsd.py

    
>  新建会话

    tmux new -s <session-name>
    
> tmux 接入会话

    tmux attach -t <session-name>

