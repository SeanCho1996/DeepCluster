# Pytorch Implementation of "Deep Clustering for Unsupervised Learning of Visual Features"
Paper source:[https://arxiv.org/pdf/1807.05520.pdf](https://arxiv.org/pdf/1807.05520.pdf)

## Requirements
torch == 1.9.0  
torchvision == 0.10.0  

(conda) faiss == 1.7.3

## Train
You first have to modify the json dict in `cfg.py`, here I used [Imagenette dataset](https://github.com/fastai/imagenette) as my test example.
You should change the values to adapt to your dataset.

Then simply run:
```shell
python train.py
```

## Predict
Once you have setup your test dataset, run:
```shell
python pred.py
```
