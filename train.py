# %% dependencies
import os
import time

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

# %% init model (and model utils)
model = vgg16(out=cfg["num_classes"])  # model = vgg16(sobel=True, out=cfg["num_classes"])
fd = int(model.top_layer.weight.size()[1])
model.top_layer = None  # Use vgg to extract features, so the classifier is discarded for now

optimizer_ft = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=30, gamma=0.1)

criterion = nn.CrossEntropyLoss()

# %% data process
image_train, label_train, image_val, label_val = image_fetch(cfg["dataset_path"])
train_dataset = ClusterDataset(image_train, label_train)

dataloader = DataLoader(train_dataset,
                        batch_size=cfg["batch_size"],
                        num_workers=8,
                        shuffle=False)

deepcluster = Kmeans(cfg["num_classes"])
cluster_log = Logger(os.path.join("./", 'clusters.log'))

# %%
def train_vgg(loader, model, crit, opt, epoch):
    """Training of the CNN.

    Args:
        loader (torch.utils.data.DataLoader): Data loader
        model (nn.Module): CNN
        crit (torch.nn): loss
        opt (torch.optim.SGD): optimizer for every parameters with True
                                requires_grad in model except top layer
        epoch (int)
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    data_time = AverageMeter()
    forward_time = AverageMeter()
    backward_time = AverageMeter()

    # switch to train mode
    model.train()

    # create an optimizer for the last fc layer
    optimizer_tl = torch.optim.SGD(
        model.top_layer.parameters(),
        lr=0.05,
        weight_decay=10**-5,
    )
    # optimizer_tl = torch.optim.Adam(filter(lambda p: p.requires_grad, model.top_layer.parameters()), lr=1e-4)

    end = time.time()
    for i, (input_tensor, target) in enumerate(loader):
        data_time.update(time.time() - end)

        # save checkpoint
        n = len(loader) * epoch + i
        if n % 25000 == 0:
            path = os.path.join(
                "./",
                'checkpoints',
                'checkpoint_' + str(n / 25000) + '.pth.tar',
            )

            print('Save checkpoint at: {0}'.format(path))
            torch.save({
                'epoch': epoch + 1,
                'arch': "vgg16",
                'state_dict': model.state_dict(),
                'optimizer' : opt.state_dict()
            }, path)

        output = model(input_tensor)
        loss = crit(output, target)

        # record loss
        losses.update(loss.item(), input_tensor.size(0))

        # compute gradient and do SGD step
        opt.zero_grad()
        optimizer_tl.zero_grad()
        loss.backward()
        opt.step()
        optimizer_tl.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i % 500) == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data: {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss: {loss.val:.4f} ({loss.avg:.4f})'
                  .format(epoch, i, len(loader), batch_time=batch_time,
                          data_time=data_time, loss=losses))

    return losses.avg


def compute_features(dataloader:DataLoader, model:nn.Module, N:int):
    """compute features using CNN feature extraction modules

    Args:
        dataloader (DataLoader): test dataset loader
        model (nn.Module): cnn model
        N (int): num classes

    Returns:
        numpy.ndarray: features obtained by CNN
    """
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

# %% train epochs
for epoch in range(200):
    end = time.time()

    # remove head
    model.top_layer = None
    model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])

    # get the features for the whole dataset
    features = compute_features(dataloader, model, len(train_dataset))

    # cluster the features
    print('Cluster the features')
    clustering_loss = deepcluster.cluster(features, verbose=True)

    # assign pseudo-labels
    print('Assign pseudo labels')
    train_dataset = cluster_assign(deepcluster.images_lists,
                                                train_dataset.image_list)

    # uniformly sample per target
    sampler = UnifLabelSampler(int(1.0 * len(train_dataset)),
                                deepcluster.images_lists)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        num_workers=8,
        sampler=sampler,
        pin_memory=True,
    )

    # set last fully connected layer
    mlp = list(model.classifier.children())
    mlp.append(nn.ReLU(inplace=True))
    model.classifier = nn.Sequential(*mlp)
    model.top_layer = nn.Linear(fd, len(deepcluster.images_lists))
    model.top_layer.weight.data.normal_(0, 0.01)
    model.top_layer.bias.data.zero_()

    # train network with clusters as pseudo-labels
    end = time.time()
    loss = train_vgg(train_dataloader, model, criterion, optimizer_ft, epoch)
    exp_lr_scheduler.step()

    # print log
    print('###### Epoch [{0}] ###### \n'
            'Time: {1:.3f} s\n'
            'Clustering loss: {2:.3f} \n'
            'ConvNet loss: {3:.3f}'
            .format(epoch, time.time() - end, clustering_loss, loss))
    try:
        nmi = normalized_mutual_info_score(
            arrange_clustering(deepcluster.images_lists),
            arrange_clustering(cluster_log.data[-1])
        )
        print('NMI against previous assignment: {0:.3f}'.format(nmi))
    except IndexError:
        pass
    print('####################### \n')
    # save running checkpoint
    torch.save({'epoch': epoch + 1,
                'arch': "VGG16",
                'state_dict': model.state_dict(),
                'optimizer' : optimizer_ft.state_dict()},
                os.path.join("./", 'checkpoint.pth.tar'))

    # save cluster assignments
    cluster_log.log(deepcluster.images_lists)
