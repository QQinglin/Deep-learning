
import torch
import numpy as np
from skimage import color
import skimage
from torch.utils.data import Dataset
from torchvision import transforms as tv_transforms


train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):
    # TODO implement the Dataset class according to the description
    def __init__(self,data,mode):
        self.data = data
        self.mode = mode
        self.transform_list = [tv_transforms.ToPILImage(),
                               tv_transforms.ToTensor(),
                               tv_transforms.Normalize(
                                   train_mean,train_std),
                               #torchvision.transforms.RandomVerticalFlip(),  # p=0.5, randomly vertically flip image
                               #torchvision.transforms.RandomHorizontalFlip(), # p=0.5, randomly horizontally flip image
                               #torchvision.transforms.RandomErasing(), # p=0.5 randomly erase some zones
                               #torchvision.transforms.RandomApply(transforms=[torchvision.transforms.ColorJitter(.5, .5)],p=.5),
                                           ] # randomly adjust brightness
        self.transform = tv_transforms.Compose(self.transform_list)

    def  __len__(self):
        return  len(self.data)

    def __getitem__(self, index):
        filename = self.data[index, 0]

        gray_image = skimage.io.imread(filename, as_gray=True)
        rgb_image = skimage.color.gray2rgb(gray_image)
        rgb_image = skimage.util.img_as_ubyte(rgb_image)

        image = self.transform(rgb_image)

        return image, torch.tensor((self.data[index, 1], self.data[index, 2]), dtype=torch.double)


