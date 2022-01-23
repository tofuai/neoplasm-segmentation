from utils import train, data, lr_scheduler
from utils.losses import UNetLoss
from models import DUNet
from torch.utils.data import DataLoader
import torch
import yaml
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main(argv):
    YAML_FP = "/home/s/polyp/neoplasm-segmentation/config/exp190122.yml"
    SAVED_FN = "DUNet_190122.pth"

    with open(YAML_FP, "r") as fp:
        config = yaml.load(fp)

    # ==================== CREATE MODEL =====================
    model = DUNet()
    loss = UNetLoss()
    # =======================================================

    # ==================== READ DATA ========================
    train_image_fps = data.read_data_file(
        config["train_txt_path"], config["train_root_dir"], format='.jpeg')
    print("Number of train images: {}".format(len(train_image_fps)))

    train_dataset = data.Dataset(
        train_image_fps,
        mask_values=config["mask_values"],
        augmentation=data.get_train_augmentation(
            height=config["height"],
            width=config["width"]
        ),
        preprocessing=data.get_preprocessing(
            data.preprocess_input
        ),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=4
    )
    # =======================================================

    # ================== TRAINING ===========================
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
        momentum=0.9
    )

    scheduler = lr_scheduler.CosineAnnealingWarmupLR(optimizer,
                                                     T_max=config["num_epoch"] -
                                                     config["warmup_epoch"],
                                                     warmup_epochs=config["warmup_epoch"],
                                                     eta_min=float(config["min_lr"]))

    train_epoch = train.TrainEpoch(
        model,
        loss=loss,
        metrics=[],
        optimizer=optimizer,
        device="cuda",
        verbose=True
    )

    saved_path = os.path.join(config["saved_dir"], SAVED_FN)

    for i in range(0, config["num_epoch"]):
        current_lr = optimizer.param_groups[0]["lr"]
        print("\nEpoch: {} - Learning Rate {}".format(i, current_lr))

        train_logs = train_epoch.run(train_loader)

        print("Save model {}".format(saved_path))
        torch.save(model.state_dict(), saved_path)

        scheduler.step()
    # =======================================================


if __name__ == "__main__":
    main(sys.argv)
