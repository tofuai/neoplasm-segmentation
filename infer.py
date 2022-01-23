from models import DUNet
from utils import data
import torch
import time
import yaml
from tqdm import tqdm
import numpy as np
import cv2
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main(argv):
    YAML_FP = "/home/s/polyp/neoplasm-segmentation/config/exp190122.yml"
    MODEL_PATH = "/home/s/polyp/models/DUNet_190122.pth"
    SAVED_RESULT = "/home/s/polyp/result/DUNet_190122"

    with open(YAML_FP) as fp:
        config = yaml.load(fp)

    # ==================== READ DATA ========================
    test_image_fps = data.read_data_file(
        config["test_txt_path"], config["test_root_dir"], format='.jpeg')
    print("Number of test images: {}".format(len(test_image_fps)))

    test_dataset = data.Dataset(
        test_image_fps,
        mask_values=config["mask_values"],
        augmentation=data.get_valid_augmentation(
            height=config["height"],
            width=config["width"]
        ),
        preprocessing=data.get_preprocessing(
            data.preprocess_input
        )
    )
    # =======================================================

    # ================ CREATE MODEL =========================
    model = DUNet()
    model.load_state_dict(torch.load(MODEL_PATH), strict=False)
    model.to(torch.device("cuda"))
    model.eval()
    # =======================================================

    # ================= INFER ===============================
    for i in tqdm(range(0, len(test_image_fps))):
        fn = test_image_fps[i].split("/")[-1].split(".")[0]
        image, label = test_dataset[i]
        image = torch.from_numpy(image).to("cuda").unsqueeze(0)

        # UNet, HarDNetMSEG
        with torch.no_grad():
            predict = model(image)
        # BlazeNeo
        # with torch.no_grad():
        #     _, predict = model(image)
        # PraNet
        # with torch.no_grad():
        #     _, _, _, predict = model(image)
        predict = torch.argmax(
            predict, dim=1, keepdims=True).squeeze().data.cpu().numpy()
        neo_predict = (predict == 0).astype(np.float)
        non_predict = (predict == 1).astype(np.float)
        output = np.zeros(
            (predict.shape[0], predict.shape[1], 3)).astype(np.uint8)
        output[neo_predict > 0] = [0, 0, 255]
        output[non_predict > 0] = [0, 255, 0]

        # NeoUNet
        # with torch.no_grad():
        #     _, _, _, predict = model(image)
        # neo_predict = predict[:, [0], :, :]
        # non_predict = predict[:, [1], :, :]

        # neo_predict = torch.sigmoid(neo_predict).squeeze().data.cpu().numpy()
        # non_predict = torch.sigmoid(non_predict).squeeze().data.cpu().numpy()

        output = np.zeros(
            (predict.shape[-2], predict.shape[-1], 3)).astype(np.uint8)
        output[(neo_predict > non_predict) * (neo_predict > 0.5)] = [0, 0, 255]
        output[(non_predict > neo_predict) * (non_predict > 0.5)] = [0, 255, 0]

        saved_path = os.path.join(SAVED_RESULT, '{}.png'.format(fn))
        cv2.imwrite(saved_path, output)
    # =======================================================


if __name__ == "__main__":
    main(sys.argv)
