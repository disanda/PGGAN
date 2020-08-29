""" Module for the data loading pipeline for the model to train """
import torch
import torchvision
from torchvision.transforms import ToTensor, Normalize, Compose, Resize, transforms
from torch.utils.data import DataLoader
import torch.utils.data as data
import os
from PIL import Image

def get_transform(new_size=None):
    """
    obtain the image transforms required for the input data
    :param new_size: size of the resized images
    :return: image_transform => transform object from TorchVision
    """
    if new_size is not None:
        image_transform = Compose([
            Resize(new_size),
            ToTensor(),
            Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
    else:
        image_transform = Compose([
            ToTensor(),
            Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
    return image_transform

class DatasetFromFolder(data.Dataset):
    def __init__(self, path, transform=None):
        super().__init__()
        self.path = path#指定自己的路径
        self.image_filenames = [x for x in os.listdir(self.path) if x.endswith('jpg') or x.endswith('png')]
        self.transform  = transform
    def __getitem__(self, index):
        a = Image.open(os.path.join(self.path, self.image_filenames[index])).convert('RGB')
        if self.transform :
            a = self.transform(a)
        return a
    def __len__(self):
        return len(self.image_filenames)

#------------test-----------
# trans = get_transform(64)
# dataset = DatasetFromFolder('/home/disanda/Desktop/dataSet/CelebAMask-HQ/CelebA-HQ-img',transform=trans)
# data = torch.utils.data.DataLoader(dataset=dataset,batch_size=64)
# for i, x in enumerate(data):
#      print(i)
#      print(x.shape)
#      x=(x+1)/2
#      torchvision.utils.save_image(x, './text1.jpg', nrow=8)
#      break
# print(type(data))
# x = iter(data)
# print(type(x))
# x = next(x)
# print(type(x))
# x=(x+1)/2
# torchvision.utils.save_image(x, './text2.jpg', nrow=8)