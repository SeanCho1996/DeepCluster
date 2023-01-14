# %%
from torch.utils.data import Dataset
import torch
import numpy as np
from tqdm import tqdm
import os
from copy import deepcopy
from PIL import Image
from glob import glob
from torchvision import transforms
from cfg import cfg

# %%
# def get_dataset(dataset_path="D:\\projects\\dataset\\imagenette2-160\\train"):
#     trans = [transforms.Resize(128),
#             # transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                 std=[0.229, 0.224, 0.225])]

#     dataset = ImageFolder(dataset_path, transform=transforms.Compose(trans))

#     return dataset

# %%
def image_fetch(train_folder:str, split_rate:float=0.9):
    image_train = []
    label_train = []
    image_val = []
    label_val = []

    # get all labels
    all_labels = os.listdir(train_folder)

    # load images from their folders
    image_folders = [os.path.join(train_folder, i) for i in all_labels]
    
    for label_i, folder in enumerate(image_folders):
        cur_list = glob(os.path.join(folder, "*.JPEG"))
        cur_label_list = [label_i for _ in range(len(cur_list))]
        split = round(len(cur_list) * split_rate)
        image_train += cur_list[:split]
        label_train += cur_label_list[:split]
        image_val += cur_list[split:]
        label_val += cur_label_list[split:]

    return image_train, label_train, image_val, label_val

# %%
class ClusterDataset(Dataset):
    def __init__(self, image_list, label_list):
        self.transform_img = transforms.Compose(
                                [transforms.Resize((cfg["input_size"], cfg["input_size"])),
                                # transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])])
        self.image_list = []

        for i in range(len(image_list)):
            self.image_list.append((image_list[i], label_list[i]))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):

        img_path, label_str = self.image_list[idx]
        img = Image.open(img_path)
        try:
            img = self.transform_img(img)
        except:
            print(img_path)
        label = torch.from_numpy(np.array(int(label_str)))

        return img, label

# %%
class PredClassDataset(Dataset):
    def __init__(self, img_list, transforms):
        super(PredClassDataset, self).__init__()

        self.img_list = img_list
        self.transforms = [transforms.Resize(cfg["input_size"]),
                           # transforms.CenterCrop(224),
                           transforms.ToTensor(),
                           transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])]
    
    def __len__(self) -> int:
        return len(self.img_list)

    def __getitem__(self, index: int):
        img = self.img_list[index]
        img = self.transforms(img)
        return img