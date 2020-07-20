import os
import argparse

import torch
import torch.nn as nn
import torch.optim as O
import torch.functional as F
import torchvision as tv

from nn_helpers import ImagePathDataset, TinyImageNetValSet, createAlexNet, train, graphTrainOutput, copyLayerWeightsExceptLast

parser = argparse.ArgumentParser(
    description='Train AlexNet on Tiny Imagenet 200.')
parser.add_argument('--data', help='dataset directory',
                    dest='data_path', default='./data/tiny-imagenet-200', type=str)
parser.add_argument('--save', help='directory to save the model',
                    dest='model_path', default='./saved_models', type=str)

# hparams
parser.add_argument('--batch-size', default=200, type=int, 
                    help='mini-batch size for training (default: 200)')
parser.add_argument('--epochs', default=200, type=int, 
                    help='number of total epochs to run (default: 200)')
parser.add_argument('--init-lr', default=0.001, type=float, 
                    help='learning rate (default: 0.001)')
parser.add_argument('--snip-factor', default=0.1, type=float, 
                    help='snip factor (default: 0.1)')
parser.add_argument('--weight-decay-rate', default=5e-4, type=float, 
                    help='set weight decay rate (default: 5e-4)')

# wandb set-up
parser.add_argument('--project', default='pretrained alexnet (SNIP)', type=str, 
                    help='name of wandb project (default: pretrained alexnet (SNIP))')  
parser.add_argument('--run-name', default='test', type=str, 
                    help='name of the run, recorded in wandb (default: test)')  

# training settings
parser.add_argument('--img-size', default=224, type=int, 
                    help='image size (default: 224)')  
parser.add_argument('--perform-snip', action='store_true',  default=False, 
                    help='perform snip (default: False)')  
parser.add_argument('--validate-every', default='512', type=int, 
                    help='validation step (default: 521)') 

# debug settings
parser.add_argument('--debug', action='store_true', default=False, 
                    help='apply general debug code (default: False)')

args = parser.parse_args()

data_path = os.path.normpath(args.data_path)
model_path = os.path.normpath(args.model_path)
os.makedirs(model_path, exist_ok=True)

image_transforms = tv.transforms.Compose([
    tv.transforms.Resize((224, 224)),
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

my_alexnet = createAlexNet()
pytorch_alexnet = tv.models.alexnet(pretrained=True)
# for transfer learning
copyLayerWeightsExceptLast(pytorch_alexnet, my_alexnet, requires_grad=False)

model_path_with_name = os.path.join(model_path, 'alexnet.pth')
o = train(args, my_alexnet, pytorch_alexnet, O.Adam, train_loader, val_loader, lr=args.lr, epochs=args.epochs,
          save=True, save_path=model_path_with_name, validate_every=args.validate_every)
graphTrainOutput(*o, epochs=args.epochs, n_samples_in_epoch=n_samples_in_epoch,
                 validate_every=validate_every)
