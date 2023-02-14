import argparse
from torch.utils.tensorboard import SummaryWriter
from monai.visualize import plot_2d_or_3d_image


parser = argparse.ArgumentParser(description="Script for pretraining generator networks")
parser.add_argument("--exp_name", type=str, default='test', help="name of the experiment")
parser.add_argument("--fold", type=int, default=0, help="fold to use for training and validation")
parser.add_argument("--log_dir", type=str, default='none', help="path to checkpoints dir")
parser.add_argument("--batch_size", type=int, default=4, help="batch size used for training")
parser.add_argument("--num_ngf", type=int, default=32, help="number of initial filters")
parser.add_argument("--num_epochs", type=int, default=50, help="number of epochs used for training")
parser.add_argument("--learn_rate", type=float, default=0.0002, help="initial learning rate used for training")
parser.add_argument("--inference_dir", type=str, default='none', help="path to inference dir")
parser.add_argument("--data_root", type=str, default='none', help="root of the train data")
parser.add_argument("--data_folder", type=str, default='none', help="folder of the train data")
parser.add_argument("--continue_train", action='store_true', help="Continue training?")
parser.add_argument("--continue_train_name", type=str, default='none', help="folder of the trained model to continue training")
opt = parser.parse_args()
# print(opt)
