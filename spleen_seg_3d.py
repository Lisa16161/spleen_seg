from monai.utils import set_determinism
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    SaveImaged,
    ScaleIntensityRanged,
    Spacingd,
    RandAffined,
    Rand3DElasticd,
    RandGaussianNoised,
    RandScaleIntensityd,
    RandGaussianSmoothd,
    RandAdjustContrastd
)
from torch.utils.tensorboard import SummaryWriter
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.config import print_config
from numpy import math
import torch
#import matplotlib.pyplot as plt
#import tempfile
#import shutil
import os
import glob
import argparse
print_config()

#Use of argparse
parser = argparse.ArgumentParser(description="Script for pretraining generator networks")
parser.add_argument("--exp_name", type=str, default='test', help="name of the experiment")
parser.add_argument("--fold", type=int, default=0, help="fold to use for training and validation")
parser.add_argument("--log_dir", type=str, default='none', help="path to checkpoints dir")
parser.add_argument("--batch_size", type=int, default=4, help="batch size used for training")
parser.add_argument("--max_epochs", type=int, default=50, help="number of epochs used for training")
parser.add_argument("--learn_rate", type=float, default=0.0002, help="initial learning rate used for training")
parser.add_argument("--inference_dir", type=str, default='none', help="path to inference dir")
parser.add_argument("--root_dir", type=str, default='none', help="root of the train data")
parser.add_argument("--data_dir", type=str, default='none', help="folder of the train data")

opt = parser.parse_args()
print(opt)

data_dir = opt.data_dir
root_dir = opt.root_dir
batch_size = opt.batch_size
max_epochs = opt.max_epochs
log_dir = opt.log_dir
exp_name = opt.exp_name
inference_dir = opt.inference_dir
train_images = sorted(glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
print(train_images)
print(os.path.join(data_dir, "imagesTr", "*.nii.gz"))

train_labels = sorted(glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))
data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(train_images, train_labels)]
train_files, test_files, val_files = data_dicts[:29], data_dicts[29:34], data_dicts[34:]


set_determinism(seed=0)

train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-57,
            a_max=164,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            # spatial_size=(96, 96, 96),
            spatial_size=(96, 96, 48),  # changed due to CBCT scan sizes
            pos=1,
            neg=1,
            num_samples=4,
            image_key="image",
            image_threshold=0,
        ),
        Rand3DElasticd(
            keys=["image", "label"],
            sigma_range=(0, 1),
            magnitude_range=(0, 1),
            spatial_size=None,
            prob=0.5,
            rotate_range=(-math.pi / 36, math.pi / 36),  # -15, 15 / -5, 5
            shear_range=None,
            translate_range=None,
            scale_range=None,
            mode=("bilinear", "nearest"),
            padding_mode="zeros",
            as_tensor_output=False
        ),
        RandGaussianNoised(
            keys=["image"],
            prob=0.5,
            mean=0.0,
            std=0.1,
            allow_missing_keys=False
        ),
        RandScaleIntensityd(
            keys=["image"],
            factors=0.05,  # this is 10%, try 5%
            prob=0.5
        ),
        RandGaussianSmoothd(
            keys=["image"],
            sigma_x=(0.25, 1.5),
            sigma_y=(0.25, 1.5),
            sigma_z=(0.25, 1.5),
            prob=0.5,
            approx='erf'
            # allow_missing_keys=False
        ),
        RandAdjustContrastd(
            keys=["image"],
            prob=0.5,
            gamma=(0.9, 1.1)
            # allow_missing_keys=False
        ),
    ]
)
val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-57,
            a_max=164,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
    ]
)

test_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-57,
            a_max=164,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
    ]
)
train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0)
# train_ds = Dataset(data=train_files, transform=train_transforms)
# use batch_size=2 to load images and use RandCropByPosNegLabeld
# to generate 2 x 4 images for network training
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0)
# val_ds = Dataset(data=val_files, transform=val_transforms)
val_loader = DataLoader(val_ds, batch_size=1)

test_ds = CacheDataset(data=test_files, transform=test_transforms, cache_rate=1.0)
# train_ds = Dataset(data=train_files, transform=train_transforms)
# use batch_size=2 to load images and use RandCropByPosNegLabeld
# to generate 2 x 4 images for network training
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = UNet(
    dimensions=3,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=Norm.BATCH,
).to(device)


loss_function = DiceLoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.Adam(model.parameters(), 1e-4)
dice_metric = DiceMetric(include_background=False, reduction="mean")

val_interval = 2
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []
post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
post_label = Compose([AsDiscrete(to_onehot=2)])

log_dir = root_dir + "models/" + exp_name + "/"

if not os.path.isdir(log_dir):
    os.makedirs(log_dir)

writer = SummaryWriter(log_dir=log_dir)

for epoch in range(max_epochs):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss = 0
    step = 0
    step = 0
    for batch_data in train_loader:
        step += 1
        inputs, labels = (
            batch_data["image"].to(device),
            batch_data["label"].to(device),
        )
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        print(f"{step}/{len(train_ds) // train_loader.batch_size}, " f"train_loss: {loss.item():.4f}")
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
    epoch_len = len(train_ds)//train_loader.batch_size
    writer.add_scalar("Train loss", epoch_loss, epoch_len * epoch + step)

    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                val_inputs, val_labels = (
                    val_data["image"].to(device),
                    val_data["label"].to(device),
                )
                roi_size = (160, 160, 160)
                sw_batch_size = 4
                val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)
                val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                # compute metric for current iteration
                dice_metric(y_pred=val_outputs, y=val_labels)

            # aggregate the final mean dice result
            metric = dice_metric.aggregate().item()

            writer.add_scalar("Val metric", metric, epoch + 1)
            # reset the status for next validation round
            dice_metric.reset()

            metric_values.append(metric)
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(log_dir, "best_metric_model.pth"))
                print("saved new best metric model")
            print(
                f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                f"\nbest mean dice: {best_metric:.4f} "
                f"at epoch: {best_metric_epoch}"
            )
# Testing loop on test data
model.eval()
test_loss, test_acc = 0, 0
with torch.inference_mode():
    for batch, (X_test, y_test) in enumerate(test_loader):
        # Send the data to target device
        X_test, y_test = X_test.to(device), y_test.to(device)
        test_pred = model(X_test)

        # 2.Calculate test loss
        loss = loss_function(test_pred, y_test)
        test_loss += loss.item()  # to get single integer from previous line

        # 3.Accuracy
        test_pred_labels = test_pred.argmax(dim=1)
        test_acc += ((test_pred_labels == y_test).sum().item() / len(test_pred_labels))

    # Divide total test loss by length of train data loader(in epoch loop)
    test_loss /= len(test_loader)

    # Calculate test accuracy per batch
    test_acc /= len(test_loader)
    print(test_loss, test_acc)

# Testing loop on train data
model.eval()
train_loss, train_acc = 0, 0
with torch.inference_mode():
    for batch, (X_train, y_train) in enumerate(train_loader):
        # Send the data to target device
        X_train, y_train = X_train.to(device), y_train.to(device)
        train_pred = model(X_train)

        # 2.Calculate train loss
        loss = loss_function(train_pred, y_train)
        train_loss += loss.item()  # to get single integer from previous line

        # 3.Accuracy
        train_pred_labels = train_pred.argmax(dim=1)
        train_acc += ((train_pred_labels == y_train).sum().item() / len(train_pred_labels))

    # Divide total test loss by length of train data loader(in epoch loop)
    train_loss /= len(train_loader)

    # Calculate test accuracy per batch
    train_acc /= len(train_loader)
