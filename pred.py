# %%
import os
import time
from shutil import rmtree, copy

import numpy as np
import torch
from sklearn.metrics.cluster import normalized_mutual_info_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from cfg import cfg
from clustering import Kmeans, arrange_clustering, cluster_assign
from dataset import ClusterDataset, image_fetch
from model import vgg16
from utils import AverageMeter, Logger, UnifLabelSampler

# %%
def compute_features(dataloader, model, N):
    print('Compute features')
    batch_time = AverageMeter()
    end = time.time()
    model.eval()
    # discard the label information in the dataloader
    for i, (input_tensor, _) in enumerate(dataloader):
        aux = model(input_tensor).data.numpy()

        if i == 0:
            features = np.zeros((N, aux.shape[1]), dtype='float32')

        aux = aux.astype('float32')
        if i < len(dataloader) - 1:
            features[i * 2: (i + 1) * 2] = aux
        else:
            # special treatment for final batch
            features[i * 2:] = aux

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i % 200) == 0:
            print('{0} / {1}\t'
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})'
                  .format(i, len(dataloader), batch_time=batch_time))
    return features

# %%
model = vgg16(out=cfg["num_classes"])
fd = int(model.top_layer.weight.size()[1])
model.top_layer = None

ckpt_path = cfg["ckpt_path"]
if os.path.isfile(ckpt_path):
    print("=> loading checkpoint '{}'".format(ckpt_path))
    checkpoint = torch.load(ckpt_path)
    # remove top_layer parameters from checkpoint
    for key in list(checkpoint['state_dict'].keys()):
        if 'top_layer' in key:
            del checkpoint['state_dict'][key]
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' "
            .format(ckpt_path))
else:
    print("=> no checkpoint found at '{}'".format(ckpt_path))

# %%
image_train, label_train, image_val, label_val = image_fetch(f"./imagenette2-160/train")
val_dataset = ClusterDataset(image_val, label_val)

dataloader = DataLoader(val_dataset,
                        batch_size=cfg["batch_size"],
                        num_workers=8,
                        shuffle=False)

# %%
deepcluster = Kmeans(10)

# %%
model.top_layer = None
model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])

# get the features for the whole dataset
features = compute_features(dataloader, model, len(val_dataset))

# cluster the features
print('Cluster the features')
_ = deepcluster.cluster(features, verbose=True)

# %%
target_folder = "./output"
for i in range(len(deepcluster.images_lists)):
    if os.path.exists(os.path.join(target_folder, f"{i}")):
        rmtree(os.path.join(target_folder, f"{i}"))
    os.mkdir(os.path.join(target_folder, f"{i}"))
    for img_idx in deepcluster.images_lists[i]:
        ori_path, _ = val_dataset.image_list[img_idx]
        copy(ori_path, os.path.join(target_folder, f"{i}"))
