# 相近参数量下Transformer模型和CNN模型在图像分类任务下的比较

## 依赖

```
torch
einops
tqdm
PIL
```

## 训练

在终端中可以采用如下的命令进行训练

```
python main.py ${cifar-100-python.tar.gz文件的完整路径}$ \
                --mode train    
                --model cnn     # 可选 cnn 或 ViT    
                -b 512          # batch_size
                -j 16           # num_workers
                -e 100          # epochs
                --lr 0.3        # learning_rate
                -o ${结果保存位置}$
```

## 测试训练好的权重

```
python main.py ${cifar-100-python.tar.gz文件的完整路径}$ \
                --mode test 
                --model ViT     # 可选 cnn 或 ViT  
                --weight ${权重地址}$
```