from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Subset
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import random

# Training settings
parser = argparse.ArgumentParser(description='PyTorch GTSRB example')
parser.add_argument('--data', type=str, default='data', metavar='D',
                    help="folder where data is located. train_data.zip and test_data.zip need to be found in the folder")
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()

torch.manual_seed(args.seed)

if torch.cuda.is_available():
    use_gpu = True
    print("Using GPU")
else:
	use_gpu = False
	print("Using CPU")

FloatTensor = torch.cuda.FloatTensor if use_gpu else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_gpu else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_gpu else torch.ByteTensor
Tensor = FloatTensor

### Data Initialization and Loading
from data import initialize_data, data_transforms,data_jitter_hue,data_jitter_brightness,data_jitter_saturation,data_jitter_contrast,data_rotate,data_hvflip,data_shear,data_translate,data_center,data_hflip,data_vflip # data.py in the same folder
initialize_data(args.data) # extracts the zip files, makes a validation set
   
# Apply data transformations on the training images to augment dataset
# train_loader = torch.utils.data.DataLoader(
#    torch.utils.data.ConcatDataset([datasets.ImageFolder(args.data + '/Train',
#    transform=data_transforms),
#    datasets.ImageFolder(args.data + '/Train',
#    transform=data_jitter_brightness),datasets.ImageFolder(args.data + '/Train',
#    transform=data_jitter_hue),datasets.ImageFolder(args.data + '/Train',
#    transform=data_jitter_contrast),datasets.ImageFolder(args.data + '/Train',
#    transform=data_jitter_saturation),datasets.ImageFolder(args.data + '/Train',
#    transform=data_translate),datasets.ImageFolder(args.data + '/Train',
#    transform=data_rotate),datasets.ImageFolder(args.data + '/Train',
#    transform=data_hvflip),datasets.ImageFolder(args.data + '/Train',
#    transform=data_center),datasets.ImageFolder(args.data + '/Train',
#    transform=data_shear)]), batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=use_gpu)
   
# Combine datasets with different augmentations
combined_train_dataset = torch.utils.data.ConcatDataset([
    datasets.ImageFolder(args.data + '/Train', transform=data_transforms),
    datasets.ImageFolder(args.data + '/Train', transform=data_jitter_brightness),
    datasets.ImageFolder(args.data + '/Train', transform=data_jitter_hue),
    datasets.ImageFolder(args.data + '/Train', transform=data_jitter_contrast),
    datasets.ImageFolder(args.data + '/Train', transform=data_jitter_saturation),
    datasets.ImageFolder(args.data + '/Train', transform=data_translate),
    datasets.ImageFolder(args.data + '/Train', transform=data_rotate),
    datasets.ImageFolder(args.data + '/Train', transform=data_hvflip),
    datasets.ImageFolder(args.data + '/Train', transform=data_center),
    datasets.ImageFolder(args.data + '/Train', transform=data_shear),
])

# Compute the subset size (10% of the combined dataset)
subset_size = int(0.1 * len(combined_train_dataset))
indices = list(range(len(combined_train_dataset)))
random.shuffle(indices)
subset_indices = indices[:subset_size]

# Create a subset dataset
train_subset = Subset(combined_train_dataset, subset_indices)

# DataLoader for the subset
train_loader = torch.utils.data.DataLoader(
    train_subset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=use_gpu
)

print(f"Training with {subset_size} samples (10% of the augmented dataset).")

# val_loader = torch.utils.data.DataLoader(
#     datasets.ImageFolder(args.data + '/val_images',
#                          transform=data_transforms),
#     batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=use_gpu)
   

# Neural Network and Optimizer
from model import Net
model = Net()

if use_gpu:
    model.cuda()

optimizer = optim.Adam(filter(lambda p: p.requires_grad,model.parameters()),lr=args.lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience=5,factor=0.5,verbose=True)

def train(epoch):
    model.train()
    correct = 0
    training_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        if use_gpu:
            data = data.cuda()
            target = target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        max_index = output.max(dim = 1)[1]
        correct += (max_index == target).sum()
        training_loss += loss
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss per example: {:.6f}\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.data.item()/(args.batch_size * args.log_interval),loss.data.item()))
    print('\nTraining set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                training_loss / len(train_loader.dataset), correct, len(train_loader.dataset),
                100. * correct / len(train_loader.dataset)))

# def validation():
#     model.eval()
#     validation_loss = 0
#     correct = 0
#     for data, target in val_loader:
#         with torch.no_grad():
#             data, target = Variable(data), Variable(target)
#             if use_gpu:
#                 data = data.cuda()
#                 target = target.cuda()
#             output = model(data)
#             validation_loss += F.nll_loss(output, target, size_average=False).data.item() # sum up batch loss
#             pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
#             correct += pred.eq(target.data.view_as(pred)).cpu().sum()

#     validation_loss /= len(val_loader.dataset)
#     scheduler.step(np.around(validation_loss,2))
#     print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#         validation_loss, correct, len(val_loader.dataset),
#         100. * correct / len(val_loader.dataset)))


for epoch in range(1, args.epochs + 1):
    train(epoch)
    # validation()
    model_file = 'selfTrained_model_' + str(epoch) + '.pth'
    torch.save(model.state_dict(), model_file)
    print('\nSaved model to ' + model_file + '. Run `python evaluate.py ' + model_file + '` to generate the Kaggle formatted csv file')
