import torch
from torch import nn
import numpy as np
import cv2
from skimage.measure import label
from scipy import ndimage as ndi
from skimage import (exposure, feature, filters, io, measure,
                     morphology, restoration, segmentation, transform,
                     util)
import skimage
import matplotlib.pyplot as plt
#import torch.nn.functional as F
import torchvision.transforms.functional as F
class MaskFilter(nn.Module):
    def __init__(self, use_cuda=False):
        super(MaskFilter, self).__init__()
        # device
        self.device = 'cuda' if use_cuda else 'cpu'

    def normalize(self, img, min_=None, max_=None):
        if min_ is None:
            min_ = img.min()
        if max_ is None:
            max_ = img.max()
        return (img - min_) / (max_ - min_)

    def get_3d_mask(self, img, min_, max_=None, th=55, width=40, label=True):
        if max_ is None:
            max_ = img.max()
            print()
        #img = torch.clamp(img, min_, max_)
        img = (255 * self.normalize(img, min_, max_)).to(torch.uint8)
        ## Remove small details
        img = F.gaussian_blur(img, kernel_size=[5, 5])
        ## Remove artifacts
        mask = torch.zeros_like(img, dtype=torch.int32)
        if not label:
            mask[img > th] = 1
            mask = morphology.binary_opening(mask.cpu().numpy(), )

            mask = morphology.remove_small_holes(
                mask,
                area_threshold=(width) ** 3)#.astype(np.int32)

        else:
            mask[img > 0] = 1
            mask = morphology.binary_opening(mask.cpu().numpy(), )

            mask = morphology.remove_small_holes(
                mask,
                area_threshold=(width) ** 3)#.astype(np.int32)

        ## Remove artifacts and small holes with binary opening
        return torch.tensor(mask, dtype=torch.float32).to(self.device)#mask.astype(np.float32)

    def forward(self, input, label=True):
        if label:
            input = torch.argmax(input, dim=1, keepdim=True)
            input = input.expand(-1, 3, -1, -1)
        B, C, H, W = input.shape
        #mask = torch.zeros((B, C, H, W)).to(self.device)
        if label:
            mask = self.get_3d_mask(input, min_=0, label=label)
        else:
            mask = self.get_3d_mask(input, min_=-1, label=label)

        return mask     # * 255


if __name__ == '__main__':  #
    for i in range(0, 606, 2):
        label = cv2.imread(f'/Users/tangwenwu/Desktop/thesis/data/train/SEG/SEG_slice_{i // 2}.png')
        image = cv2.imread(f'/Users/tangwenwu/Desktop/thesis/data/train/CT/CT_slice_{i // 2}.png')
        maskfilter = MaskFilter()
        mask_label = maskfilter(label, label=True)
        mask_image = maskfilter(image, label=False)
        # cv2.imwrite(f'/Users/tangwenwu/Desktop/thesis/data/train/check/slice_{i//3}.png', image)
        cv2.imwrite(f'/Users/tangwenwu/Desktop/thesis/data/train/check/slice_{i}.png', mask_image)  # image_mask 偶数
        cv2.imwrite(f'/Users/tangwenwu/Desktop/thesis/data/train/check/slice_{i + 1}.png', mask_label)  # label_mask 奇数
