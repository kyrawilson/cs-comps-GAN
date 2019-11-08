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
# Needed for some reason to make CUDA work
torch.cuda.current_device()
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from tqdm import tqdm
import uuid
import torch.utils.data as data

from data import ImgCaptionData
from model import Generator

# Defines some backend for matplotlib to use or something
matplotlib.use('agg')
import matplotlib.pyplot as plt

def parse_args():
    ''' Uses argparse to parse command line args. Returns args, a namespace variable
    (contains all of the variables specified below) - e.g. args.bsize.
    Dashes get converted to underscores automatically.'''
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--bsize', type=int, default=64,
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
    parser.add_argument('--img-rep-channels', type=int, default=512,
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
    parser.add_argument('--square-momentum', type=float, default=0.999,
                        help='momentum to use for squared gradient in Adam')
    parser.add_argument('--text-rep-dim', type=int, default=128,
                        help='size of the text representation')

    # Argument post-processing
    args = parser.parse_args()
    # Only use CUDA if torch won't get angry
    args.cuda = args.cuda and torch.cuda.is_available()

    return args

def set_seeds(seed):
    """ Set random seeds to ensure result reproducibility.
    """
    # If a seed was not given by command line arg, make one randomly
    if seed is None:
        seed = random.randrange(2**32)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # May be unnecessary, but is left here just to be safe
    torch.cuda.manual_seed_all(seed)
    return seed

def make_model_dir(out_dir):
    ''' Make a directory for results and parameters of the current model, 
    using a randomly generated identifier'''
    # Model IDs
    ### Models are stored in ./saves/models, in folders with their id as the name
    # Generates a random id - uuid stands for Universally Unique IDentifier
    # uuid4 specifies that it's version 4
    model_id = str(uuid.uuid4())
    # If the model directory doesn't exist (hopefully it doesn't!) make it
    if not os.path.isdir(os.path.join(out_dir, model_id)):
        os.makedirs(os.path.join(out_dir, model_id))
    else:
        raise RuntimeException("Generated identical model IDs?")
    # Return both the model ID and the path to the model dir in order to
    # output the ID in the params.json file
    return model_id, os.path.join(out_dir, model_id)

def make_kwargs(args, seed, model_id):
    ''' Makes kwargs dictionary, both for passing to nn modules and
    for outputting as .json (for replicability). Mostly just command-line
    args.'''

    # For descriptions of each variable, look at the argparse help string
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
        'img_rep_channels': args.img_rep_channels,
        'model_id': model_id,
        'momentum': args.momentum,
        'seed': seed,
        'schedule_epochs': args.schedule_epochs,
        'schedule_gamma': args.schedule_gamma,
        'square_momentum': args.square_momentum,
        'text_rep_dim': args.text_rep_dim
    }

    # Dump kwargs to model folder
    with open(os.path.join(args.out_dir, model_id, 'params.json'), 
              'w') as params_f:
        # indent: when set to something besides None, enables pretty-printing
        # of json file; the specific integer sets the tab size in num. spaces
        json.dump(kwargs, params_f, indent=2)

    # These transforms are modifications we make to the images when
    # we train on them: making these modifications helps the network
    # to learn to ignore insignificant changes to the images (to avoid
    # overfitting)
    # TODO make this a command line arg
    # TODO make img size a global constant
    kwargs['img_transform'] = transforms.Compose([transforms.Resize((136, 136)),
                                                 transforms.RandomCrop((128, 128)),
                                                 transforms.RandomRotation(10),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.ToTensor()])
    if args.cuda:
        kwargs['device'] = torch.device('cuda')
    else:
        kwargs['device'] = torch.device('cpu')

    return kwargs

def train(G, epoch, loader, optimizer, val=False):
    """ Train (or validate) models for a single epoch.
    G: generator object
    epoch: epoch number
    loader: the training or validation data loader
    """
    val = optimizer == None
    # train_loader.init_epoch()
    # pbar = tqdm(total=len(train_loader))
    # Sets model in training mode
    G.train()
    total_loss = 0
    pbar = tqdm(total=len(train_loader))
    for batch_idx, batch in enumerate(train_loader):
        if not val:
            optimizer.zero_grad()
        img = batch[0].to(kwargs['device'])
        # text = batch[1].to(kwargs['device'])
        text = batch[1]
        fake = G(img, text)
        # Measures dissimilarity between decoded image and input
        # Need to instantiate the loss fn - it's an object, not a function
        lossFn = nn.MSELoss()
        loss = lossFn(fake, img)
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
            avg_loss = total_loss/((batch_idx+1)*img.shape[0])
            print()
            print(type + ' epoch {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * img.shape[0], len(train_loader.dataset),
                100. * batch_idx / len(train_loader), avg_loss))
        pbar.update()
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, avg_loss))
    pbar.close()
    return avg_loss

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

if __name__ == "__main__":
    args = parse_args()
    seed = set_seeds(args.seed)
    model_id, model_dir = make_model_dir(args.out_dir)
    kwargs = make_kwargs(args, seed, model_id)
    # ImgCaptionData returns a torch.data.Dataset object, containing images,
    # captions and class labels (i.e. type of bird)
    # ImgCaptionData uses kwargs for the data directory as well as for
    # determining which transformations to apply to the images (rotation, etc.)
    train_data = ImgCaptionData(**kwargs)
    # The dataloader takes the data in our Dataset object, shuffles it,
    # and then turns it into a batch iterator
    train_loader = data.DataLoader(train_data, 
                                   batch_size=args.bsize,
                                   shuffle=True)

    # TODO what is our validation split?
    # val_data = ???
    # val_loader = data.DataLoader(val_data,
                                    # batch_size=args.batch_size,
                                    # shuffle=True)
    val_loader = [(0, 0)]
    # Store the training (first column) and val (first column) losses
    # for each epoch, in order to graph them later
    losses = np.zeros((args.epochs, 2))
    # Make a generator object, using kwargs to pass in hyperparameters 
    G = Generator(**kwargs)
    # TODO maybe only get parameters in G that require gradients?
    # Instantiate an "Adam" type optimizer over G's parameters (i.e. weights)
    # To determine how quickly to learn at any given time, Adam sums together
    # a moving average of the gradients of previous epochs and the squared gradients
    # The gradients of previous epochs decay exponentially by a factor of
    # args.momentum (for non-squared gradients) and args.square_momentum (for
    # squared gradients)
    optim_G = optim.Adam(G.parameters(), 
                         lr=0.002, 
                         betas=[args.momentum, args.square_momentum])
    for epoch in range(args.epochs):
        # train generator
        avg_train_loss = train(G, epoch, train_loader, optim_G)
        losses[epoch][0] = avg_train_loss

        # test generator
        # avg_test_loss = train(G, epoch, train_loader, None)
        # losses[1][epoch] = avg_test_loss
    dest = os.path.join(model_dir, 'loss.png')
    # Outputs a plot of the losses to dest
    plot_losses(losses, dest)
