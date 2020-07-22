import os
import argparse

import torch
import torch.nn as nn
import torch.optim as O
import torch.functional as F
import torchvision as tv

from nn_helpers import ImagePathDataset, TinyImageNetValSet, createAlexNet, train, graphTrainOutput, copyLayerWeightsExceptLast

# hparams
#hparams = {'batch_size':200,
#           'epochs':5,
#           'init_lr':0.001,
#           'snip_factor':1.0,
#           'weight_decay_rate':5e-4}

parser = argparse.ArgumentParser(
    description='Train AlexNet on Tiny Imagenet 200.')
parser.add_argument('--data', help='dataset directory',
                    dest='data_path', default='./data/tiny-imagenet-200', type=str)
parser.add_argument('--save', help='directory to save the model',
                    dest='model_path', default='./saved_models', type=str)

# hparams
parser.add_argument('--batch_size', default=200, type=int, 
                    help='mini batch size for training (default: 200)')
parser.add_argument('--epochs', default=5, type=int, 
                    help='number of total epochs to run (default: 5)')
parser.add_argument('--init_lr', default=0.001, type=float, 
                    help='learning rate (default: 0.001)')
parser.add_argument('--snip_factor', default=1.0, type=float, 
                    help='snip factor (default: 1.0)')
parser.add_argument('--weight_decay_rate', default=5e-4, type=float, 
                    help='set weight decay rate (default: 5e-4)')

# wandb set-up
parser.add_argument('--project', default='pretrained_alexnet_SNIP', type=str, 
                    help='name of wandb project (default: pretrained alexnet (SNIP))')  
parser.add_argument('--run_name', default='test', type=str, 
                    help='name of the run, recorded in wandb (default: test)')  

# training settings
parser.add_argument('--img_size', default=224, type=int, 
                    help='image size (default: 224)')  
parser.add_argument('--perform_snip', action='store_true',  default=False, 
                    help='perform snip (default: False)')  
parser.add_argument('--validate_every', default='512', type=int, 
                    help='validation step (default: 521)') 
parser.add_argument('--topk', default=5, type=int, 
                    help='top-k accuracy to get besides 1 (default: 5)')
parser.add_argument('--save_model', action='store_true', default=False, 
                    help='save the best version of the model (default: False)')
parser.add_argument('--cuda', default=2, type=int, 
                    help='cuda device number to use (default: 2)')
parser.add_argument('--acc_target', default=35.0, type=float, 
                    help='target validation accuracy to stop training at (default: 35.0)')
parser.add_argument('--tl', action='store_true', default=False, 
                    help='train only output layer (default: False)')


# debug settings
parser.add_argument('--debug', action='store_true', default=False, 
                    help='apply general debug code (default: False)')

args = parser.parse_args()

data_path = os.path.normpath(args.data_path)
model_path = os.path.normpath(args.model_path)
os.makedirs(model_path, exist_ok=True)

image_transforms = tv.transforms.Compose([
    tv.transforms.Resize((args.img_size, args.img_size)),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

train_dataset = tv.datasets.ImageFolder(os.path.join(
    data_path, 'train'), transform=image_transforms)
train_loader = torch.utils.data.DataLoader(
    train_dataset, shuffle=True, batch_size=args.batch_size)

val_dataset = TinyImageNetValSet(os.path.join(
    data_path, 'val'), transform=image_transforms)
val_loader = torch.utils.data.DataLoader(
    val_dataset, shuffle=False, batch_size=1000)

n_samples_in_epoch = len(train_loader)

model_path_with_name = os.path.join(model_path, 'alexnet.pth')
o = train(args, O.Adam, train_loader, val_loader,
          save=True, save_path=model_path_with_name)
#graphTrainOutput(*o, epochs=args.epochs, n_samples_in_epoch=n_samples_in_epoch,
#                 validate_every=args.validate_every)
