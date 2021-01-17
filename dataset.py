# -*- coding: <encoding name> -*-
"""
crowd counting dataset
"""
from torch.utils.data import Dataset
from PIL import Image
import os
import glob
import numpy as np
import h5py
import cv2
import random
from torchvision.transforms import functional


################################################################################
# crowd counting dataset
################################################################################
class CrowdCountingDataset(Dataset):
    def __init__(self, dir_path, transforms, scale=8, mode='train'):
        """
        :param
             dir_path(str) -- the path of the image directory
             transforms --
             scale(int) -- density map scale factor
             mode(str) --
        """
        self.transforms = transforms
        self.scale = scale
        self.mode = mode

        # acquire image path
        self.img_paths = []
        for img_path in glob.glob(os.path.join(dir_path, '*.jpg')):
            self.img_paths.append(img_path)

    def __getitem__(self, index):
        ##--load image--##
        img_path = self.img_paths[index]
        # read image
        img = Image.open(img_path).convert('RGB')
        # image size
        img_width, img_height = img.size

        ##--load density map--##
        density_path = img_path.replace('.jpg', '.h5').replace('images', 'density')
        # read density map
        with h5py.File(density_path, 'r') as hf:
            density = np.asarray(hf['density'])

        if self.mode != 'train':
            # image
            img = self.transforms(img)
            # density map
            gt = np.sum(density)
            density = cv2.resize(density,
                                 (density.shape[1] // self.scale, density.shape[0] // self.scale),
                                 interpolation=cv2.INTER_CUBIC) * (self.scale ** 2)
            density = density[np.newaxis, :, :]

            return img, gt, density

        # random resize
        short = min(img_width, img_height)
        if short < 512:
            scale = 512 / short
            img_width = round(img_width * scale)
            img_height = round(img_height * scale)
            img = img.resize((img_width, img_height), Image.BILINEAR)
            density = cv2.resize(density, (img_width, img_height), interpolation=cv2.INTER_LINEAR) / scale / scale
        scale = random.uniform(0.8, 1.2)
        img_width = round(img_width * scale)
        img_height = round(img_height * scale)
        img = img.resize((img_width, img_height), Image.BILINEAR)
        density = cv2.resize(density, (img_width, img_height), interpolation=cv2.INTER_LINEAR) / scale / scale

        # random crop
        h, w = 400, 400
        dh = random.randint(0, img_height - h)
        dw = random.randint(0, img_width - w)
        img = img.crop((dw, dh, dw + w, dh + h))
        density = density[dh:dh + h, dw:dw + w]

        # random flip
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            density = density[:, ::-1]

        # random gamma
        if random.random() < 0.3:
            gamma = random.uniform(0.5, 1.5)
            img = functional.adjust_gamma(img, gamma)

        # random to gray
        if random.random() < 0.1:
            img = functional.to_grayscale(img, num_output_channels=3)

        img = self.transforms(img)
        density = cv2.resize(density, (density.shape[1] // self.scale, density.shape[0] // self.scale),
                             interpolation=cv2.INTER_LINEAR) * self.scale * self.scale
        density = density[np.newaxis, :, :]

        return img, density

    def __len__(self):
        return len(self.img_paths)
