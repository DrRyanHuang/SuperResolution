import numpy as np
import os
from PIL import Image
import torch
from torchvision import transforms
from torch.autograd import Variable
import numpy as np


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    """
    TO-DO: Skip grayscale images or return error
    """
    img = Image.open(filepath)
    return img


class DatasetFromFolder(torch.utils.data.Dataset):
    def __init__(self, image_dir, input_transform=None, target_transform=None):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [os.path.join(image_dir, x) for x in os.listdir(image_dir) if is_image_file(x)]

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        input_ = load_img(self.image_filenames[index])
        if input_.mode == 'L':
            # 灰度图自动转化为RGB三通道图片
            input_ = input_.convert("RGB")
        if input_.mode not in ['L', 'RGB']:
            raise ValueError("FUCKING DATASET : {}".format(self.image_filenames[index]))
        target = input_.copy()
        if self.input_transform is not None:
            input_ = self.input_transform(input_)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return input_, target

    def __len__(self):
        return len(self.image_filenames)


def calculate_valid_crop_size(crop_size, upscale_factor):
    # 把它整成可以整除的亚子
    return crop_size - (crop_size % upscale_factor)


def img_transform(crop_size, upscale_factor=1):
    return transforms.Compose([
        transforms.Resize(crop_size // upscale_factor), # smaller edge of the image will be matched to this number
        transforms.CenterCrop(crop_size // upscale_factor), # 进行中心裁剪
        transforms.ToTensor()])


def get_training_set(path, hr_size=256, upscale_factor=4):
    train_dir = path
    crop_size = calculate_valid_crop_size(hr_size, upscale_factor)

    return DatasetFromFolder(train_dir,
                             input_transform=img_transform(crop_size, upscale_factor=upscale_factor),
                             target_transform=img_transform(crop_size))


def denorm_meanstd(arr, mean, std):
    new_img = np.zeros_like(arr)
    for i in range(3):
        new_img[i, :, :] = arr[i, :, :] * std[i]
        new_img[i, :, :] += mean[i]
    return new_img


def image_loader(image_name, max_sz=256):
    """ forked from pytorch tutorials """
    r_image = Image.open(image_name)
    mindim = np.min((np.max(r_image.size[:2]), max_sz))

    loader = transforms.Compose([transforms.CenterCrop(mindim),
                                 transforms.ToTensor()])

    image = Variable(loader(r_image))

    return image.unsqueeze(0) # 最后再加一维
