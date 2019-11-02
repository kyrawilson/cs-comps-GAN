''' 
File: train.py
Author: Will Schwarzer (schwarzerw@carleton.edu)
Date: October 25, 2019
Loads the Caltech Birds (CUB) dataset, instantiates a replicated version of
TAGAN, then trains it on the dataset.

For now, just instantiates the generator, and trains it to reproduce the input
image.

Some simple utility code is reused from another personal research project.
'''

import json
import matplotlib
import numpy as np
import os
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import tqdm
import uuid
import torch.utils.data as data

from data import ImgCaptionData
from model import Generator

# from models import Generator

matplotlib.use('agg')
import matplotlib.pyplot as plt

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--bsize', type=int, default=10,
                        help='size of training and testing batches')
    parser.add_argument('--caption-files', type=str, default='text_c10',
                        help='name of folder containing image captions')
    parser.add_argument('--classes-file', type=str, default='classes.txt',
                        help='name of file containing list of bird classes')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--data', type=str, default='mini_TAGAN_data',
                        help='folder of data to use')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train for')
    parser.add_argument('--img-files', type=str, default='images',
                        help='name of image folder in data')
    parser.add_argument('--img-rep-dim', type=int, default=512,
                        help='size of the image representation')
    parser.add_argument('--lr', type=float, default=2e-3, metavar='LR',
                        help='learning rate')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='momentum of Adam optimizer')
    parser.add_argument('--out-dir', type=str, default='./saves/',
                        help='where to save models')
    parser.add_argument('--schedule-gamma', type=float, default=0.5,
                        help='factor to reduce lr by on schedule')
    parser.add_argument('--schedule-epochs', type=int, default=100,
                        help='number of epochs to reduce lr after')
    parser.add_argument('--seed', type=int, default=None,
                        help='random seed to use')
    parser.add_argument('--text-rep-dim', type=int, default=128,
                        help='size of the text representation')

    # Argument post-processing
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    return args

def plot_losses(losses, dest):
    '''
    Plot losses over time.
    '''
    plt.figure()
    plt.plot(range(len(losses)), losses[:, 0], '-', label='train')
    plt.plot(range(len(losses)), losses[:, 1], '-', label='val')
    plt.tight_layout()
    plt.legend()
    plt.savefig(dest)

def make_kwargs(args, seed, model_id):
    kwargs = {
        'bsize': args.bsize,
        'caption_files': os.path.join(args.data, args.caption_files),
        'classes_file': os.path.join(args.data, args.classes_file),
        'cuda': args.cuda,
        'data': args.data,
        'date:': time.strftime("%Y-%m-%d %H:%M"),
        'epochs': args.epochs,
        'lr': args.lr,
        'img_files': os.path.join(args.data, args.img_files),
        'img_rep_dim': args.img_rep_dim,
        'model_id': model_id,
        'momentum': args.momentum,
        'seed': seed,
        'schedule_epochs': args.schedule_epochs,
        'schedule_gamma': args.schedule_gamma,
        'text_rep_dim': args.text_rep_dim
    }

    with open(os.path.join(args.out_dir, model_id, 'params.json'), 
              'w') as params_f:
        # indent: when set to something besides None, enables pretty-printing
        # of json file; the specific integer sets the tab size in num. spaces
        json.dump(kwargs, params_f, indent=2)

    # TODO make this a command line arg
    # TODO make img size a global constant
    kwargs['img_transform'] = transforms.Compose([transforms.Resize((136, 136)),
                                                 transforms.RandomCrop((128, 128)),
                                                 transforms.RandomRotation(10),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.ToTensor()])

    return kwargs

def set_seeds(seed):
    """ Set random seeds to ensure result reproducibility.
    """
    if seed is None:
        seed = random.randrange(2**32)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # May be unnecessary, but is left here just to be safe
    torch.cuda.manual_seed_all(seed)
    return seed

def make_model_dir(out_dir):
    # Model IDs
    ### Models are stored in ./saves/models, in folders with their id as the name
    # Generates a random id
    model_id = str(uuid.uuid4())
    if not os.path.isdir(os.path.join(out_dir, model_id)):
        os.makedirs(os.path.join(out_dir, model_id))
    return model_id, os.path.join(out_dir, model_id)

def train(G, epoch, loader, optimizer, val=False):
    """ Train (or validate) models for a single epoch.
    """
    val = optimizer == None
    train_loader.init_epoch()
    pbar = tqdm(total=len(train_loader))
    # Sets model in training mode
    model.train()
    for batch_idx, batch in enumerate(train_loader):
        if not val:
            optimizer.zero_grad()
        img = batch[0]
        text = batch[1]
        fake = G(img)
        # Measures dissimilarity between decoded image and input
        loss = nn.MSELoss(fake, img)
        total_loss += loss
        if not val:
            loss.backward()
            optimizer.step()
        # Update progress
        if batch_idx % args.log_interval == 0:
            if val:
                type = 'Val'
            else:
                type = 'Train'
            avg_loss = total_loss/(batch_idx*img.shape[0])
            print(type + ' Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * img.shape[0], len(train_loader.dataset),
                100. * batch_idx / len(train_loader), avg_loss))
        pbar.update()
    pbar.close()
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, avg_loss))
    return avg_loss

if __name__ == "__main__":
    args = parse_args()
    seed = set_seeds(args.seed)
    model_id, model_dir = make_model_dir(args.out_dir)
    kwargs = make_kwargs(args, seed, model_id)
    train_data = ImgCaptionData(**kwargs)
    train_loader = data.DataLoader(train_data, 
                                   batch_size=args.bsize,
                                   shuffle=True)
    # train_loader = [(0, 0)]

    # val_data = ???
    # val_loader = data.DataLoader(val_data,
                                    # batch_size=args.batch_size,
                                    # shuffle=True)
    val_loader = [(0, 0)]
    # one row of losses for training, one for testing
    losses = np.zeros((args.epochs, 2))
    G = Generator()
    # TODO maybe only get parameters in G that require gradients?
    # optim_G = optim.Adam(G.parameters(), lr=args.lr, weight_decay=1e-4)
    for epoch in range(args.epochs):
        # train generator
        optim_G = optim.Adam(G.parameters(), lr=0.002, momentum=0.5)
        avg_train_loss = train(G, epoch, train_loader, optim_G)
        losses[0][epoch] = avg_train_loss

        # test generator
        # avg_test_loss = train(G, epoch, train_loader, None)
        # losses[1][epoch] = avg_test_loss
    dest = os.path.join(model_dir, 'loss.png')
    plot_losses(losses, dest)
