# Pytorch Implementation of "Deep Clustering for Unsupervised Learning of Visual Features"
Paper source: [https://arxiv.org/pdf/1807.05520.pdf](https://arxiv.org/pdf/1807.05520.pdf)

## Requirements 依赖
torch == 1.9.0  
torchvision == 0.10.0  
numpy == 1.23.5  
scikit-learn == 1.2.0  
scipy == 1.10.0  
tqdm == 4.6.14  
pillow == 9.0.4

(conda) faiss == 1.7.3

## Dataset format 数据集格式
The current dataset format is set as follows. You don't have to strictly follow this example, if you are using a different format, you just have to modify the `ClusterDataset` in `dataset.py` to load them in your way.

当前的数据集格式设置如下。其实不必严格按照这个规则，如果想使用不同的格式，你只需要修改`dataset.py`中的`ClusterDataset`，以你的方式加载图像就可以。

    DATASET_NAME/
        ├── class_name1/  ## 文件夹的名即为类名
        │   ├── xxx.JPEG  ## 图像文件
        │   ├── xxx.JPEG
        │   ├── xxx.JPEG
        │   ├── xxx.JPEG
        │   ├── xxx.JPEG
        │   └── ...
        ├── class_name2/
        │   ├── xxx.JPEG
        │   ├── xxx.JPEG
        │   ├── xxx.JPEG
        │   ├── xxx.JPEG
        │   ├── xxx.JPEG
        │   └── ...
        ├── class_name3/
        ├── class_name4/
        └── ...

## Train 训练
You first have to modify the json dict in `cfg.py`, here I used [Imagenette dataset](https://github.com/fastai/imagenette) as my test example.
You should change the values to adapt to your dataset.

首先需要修改`cfg.py`中的json dict，这里我用[Imagenette dataset](https://github.com/fastai/imagenette)作为我的测试例子，你可能需要改变字典里的一些值以适应其他数据集。

Then simply run：  
然后直接运行:
```shell
python train.py
```

## Predict 预测
Once you have setup your test dataset, run:  
设置好了测试数据集之后，运行：  
```shell
python pred.py
```

## Notes 备注
The default model is adapted to train and predict on a **CPU** device, if you wish to train the model on **CUDA** device, use `train_cuda.py` and `pred_cuda.py`.  
Some further improvements may be done by modifying the following params:  
* Enable "sobel" in the model:  `model = vgg16(sobel=True, out=cfg["num_classes"])`  (Do this in both `train.py` and `pred.py`)  
* Try tuning learning rate after each epoch by allowing `exp_lr_scheduler.step()` (Even I have already written this in my code, this is not applied in my initial training, I added it after the training started.), maybe also try set lr_scheduler as `ReduceOnPlateau` while monitoring training loss.  
* Use `Adam` optimizer for both feature extraction (this is already been set in the code) and the classifier (classifier uses SGD for the time being)
* Try using 'PIC' (Power Iteration Clustering) as the clustering method. (I did not implement this setting in my code, have to check the [original repo](https://github.com/facebookresearch/deepcluster/blob/2d1927e8e3dd272329e879e510fbbdf1b1d02d17/main.py#L37) to enable it)

默认的模型适应于在**CPU**设备上训练和预测，如果需要在**CUDA**设备上训练模型，使用`train_cuda.py`和`pred_cuda.py`。  
一些进一步的改进可以通过修改以下参数来完成。  
* 在模型中启用 "sobel"算子。 设置`model = vgg16(sobel=True, out=cfg["num_classes"])` (需要在`train.py`和`pred.py`中都这样做)  
* 尝试使用`exp_lr_scheduler.step()`在每个epoch结束后来调整学习率（我已经在代码中写了这个部分，但我当前的测试没有使用这个设置，我是在训练开始后来才添加的），也许还可以尝试将lr_scheduler设置为`ReduceOnPlateau`监测train loss来调整学习率。  
* 对特征提取（代码中已经设置好了）和分类器（分类器暂时使用SGD）都使用`Adam`优化器。
* 尝试使用 "PIC"（Power Iteration Clustering）作为聚类方法。(我没有在现在的代码中实现这个配置，需要学习一下[原始repo](https://github.com/facebookresearch/deepcluster/blob/2d1927e8e3dd272329e879e510fbbdf1b1d02d17/main.py#L37)来启用它)

## Discussion 讨论
Theoretically 4096 dimensions are enough to characterize an image, is it possible that the pseudo-labels assigned for each epoch re-clustering are different, causing the part of the VGG network that extracts features to be adjusted all the time, which in turn causes the features to be clustered to change all the time.

A potential improvement is that training the VGG on a labeled dataset, and directly use it in the feature extraction step without tuning it at all.

理论上来讲，4096个维度足以描述一幅图像的特征，是否有可能为每个epoch重新聚类分配的伪标签是有差异的，导致VGG网络中提取特征的部分一直在被调整，这又进一步导致要聚类的特征一直在变化。

一个可能的改进是，在一个有标签的数据集上训练一个完整的VGG，然后直接在特征提取这一步里使用这个VGG的特征提取部分，把它deteach掉，不需要调整它的参数。