import torch
import os
import numpy as np
from glob import glob
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class toBinary(object):
    def __call__(self, label):
        label = np.array(label)
        # print(image)
        label = label * (label > 127)
        label = Image.fromarray(label)
        return label


transform_image = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    # transforms.Normalize((0.4951, 0.4956, 0.4972), (0.1734, 0.1750, 0.1736)),
])

transform_label = transforms.Compose([
    transforms.Grayscale(),
    toBinary(),
    transforms.ToTensor(),
    # transforms.Normalize((0.4938, 0.4933, 0.4880), (0.1707, 0.1704, 0.1672)),
])


class CellDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1, transform=None, target_transform=None):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        self.ids = [os.path.splitext(item)[0] for item in os.listdir(imgs_dir) if not item.startswith('.')]

    def __len__(self):
        return len(self.ids)

    # @classmethod
    # def preprocess(cls, pil_img, scale):
    #     w, h = pil_img.size
    #     newH, newW = int(scale * h), int(scale * w)
    #     pil_img = pil_img.resize((newW, newH))
    #     img_nd = np.array(pil_img)
    #     if len(img_nd.shape) == 2:
    #         img_nd = np.expand_dims(img_nd, axis=2)
    #     # HWC to CHW
    #     img_trans = img_nd.transpose((2, 0, 1))  # channel 前置
    #     if img_trans.max() > 1:
    #         img_trans = img_trans / 255
    #     return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + '*')
        img_file = glob(self.imgs_dir + idx + '*')
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])
        img = transform_image(img)
        mask = transform_label(mask)
        # img = self.preprocess(img, self.scale)
        # mask = self.preprocess(mask, self.scale)
        return {
            "image": img,
            "mask": mask
        }


