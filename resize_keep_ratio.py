import os
import numpy as np
import cv2
import albumentations as albu
from tqdm import tqdm
from glob import glob


def main():
    SAVED_DIR = "/home/s/NeoDataset-1300-440/label_images/"
    list_image_fp = glob("/home/s/NeoDataset-1300/label_images/*.png")

    for image_path in tqdm(list_image_fp):
        fn = image_path.split("/")[-1]
        
        image = cv2.imread(image_path)

        transform = albu.Compose([
            albu.SmallestMaxSize(max_size=440, interpolation=cv2.INTER_NEAREST, always_apply=True),
        ])

        # Augment an image
        transformed = transform(image=image)
        image = transformed["image"]

        saved_path = os.path.join(SAVED_DIR, fn)
        cv2.imwrite(saved_path, image)


if __name__ == "__main__":
    main()