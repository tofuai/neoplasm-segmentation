import cv2
import os
import sys
import numpy as np
import albumentations as albu
import matplotlib.pyplot as plt
from torch.utils.data import Dataset as BaseDataset


def preprocess_input(
    x,
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
    input_space="RGB",
    input_range=[0, 1],
    **kwargs
):

    if input_space == "BGR":
        x = x[..., ::-1].copy()

    if input_range is not None:
        if x.max() > 1 and input_range[1] == 1:
            x = x / 255.0

    if mean is not None:
        mean = np.array(mean)
        x = x - mean

    if std is not None:
        std = np.array(std)
        x = x / std

    return x


def read_data_file(data_fp, root_dir, format=''):
    image_fps = []

    if isinstance(data_fp, list):
        for x in data_fp:
            with open(x, "r+") as fp:
                lines = fp.readlines()
                for i in range(0, len(lines)):
                    image_file_path = lines[i].rstrip('\n').strip()
                    image_file_path = os.path.join(root_dir, image_file_path)
                    image_fps.append(image_file_path)
    else:
        with open(data_fp, "r+") as fp:
            lines = fp.readlines()
            for i in range(0, len(lines)):
                image_file_path = lines[i].rstrip('\n').strip()
                image_file_path = os.path.join(root_dir, image_file_path)
                image_fps.append(image_file_path)

    image_fps = [x + format for x in image_fps]

    return image_fps


class Dataset(BaseDataset):
    def __init__(
        self,
        image_fps,
        mask_values=None,
        augmentation=None,
        preprocessing=None,
    ):
        self.image_fps = image_fps
        self.mask_fps = [x.replace('images', 'label_images').replace(
            '.jpeg', '.png') for x in image_fps]

        self.mask_values = mask_values

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        # read data
        image = cv2.imread(self.image_fps[i])
        if image is None:
            print("Can not read image file {}".format(self.image_fps[i]))
            sys.exit(1)

        mask = cv2.imread(self.mask_fps[i])
        if mask is None:
            print("Can not read mask file {}".format(self.mask_fps[i]))
            sys.exit(1)

        # Convert BGR format to RGB format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        # Extract mask for each classes
        neo = np.all(
            mask == self.mask_values["neoplastic"], axis=-1).astype('float')
        non = np.all(
            mask == self.mask_values["non-neoplastic"], axis=-1).astype('float')
        ignore = np.all(
            mask == self.mask_values["undefined"], axis=-1).astype('float')
        ignore_2 = np.all(mask == [255, 255, 255], axis=-1).astype('float')
        ignore = ignore + ignore_2

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, masks=[neo, non, ignore])
            image, masks = sample["image"], sample["masks"]
            neo, non, ignore = masks

        # Create output 2 which segments neoplastic, non-neoplastic and background
        output = np.stack([ignore, neo, non], axis=-1)
        background = 1 - output.sum(axis=-1, keepdims=True)
        output = np.concatenate((output, background), axis=-1)

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=output)
            image, output = sample["image"], sample["mask"]

        return image, output

    def __len__(self):
        return len(self.image_fps)


def get_train_augmentation(height, width):
    train_transform = [
        albu.Resize(height=height, width=width,
                    interpolation=cv2.INTER_NEAREST, always_apply=True),
        albu.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1,
                              rotate_limit=10, border_mode=0, p=0.5),
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.ColorJitter(brightness=0.2, contrast=0.2,
                         saturation=0.2, hue=0.2, p=0.5),
        albu.MotionBlur(blur_limit=(3, 7), p=0.5)
    ]
    return albu.Compose(train_transform, p=1)


def get_valid_augmentation(height, width):
    test_transform = [
        albu.Resize(height=height, width=width,
                    interpolation=cv2.INTER_NEAREST, always_apply=True)
    ]
    return albu.Compose(test_transform, p=1)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


def visualize(saved_path, **images):
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    if saved_path is not None:
        plt.savefig(saved_path)
        plt.close()
    else:
        plt.show()
