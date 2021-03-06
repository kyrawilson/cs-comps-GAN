'''
File: train.py
Author: Will Schwarzer (schwarzerw@carleton.edu)
Date: November 4, 2019
Loads the Caltech Birds (CUB) dataset, instantiates a replicated version of
TAGAN, then trains it on the dataset.

Some simple utility code is reused from another personal research project.
'''

import json
import matplotlib
import numpy as np
import os
import random
import time
import torch
torch.cuda.current_device()
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from tqdm import tqdm
import uuid
import torch.utils.data as data
import torch.nn.functional as F

from data import ImgCaptionData
from model import Generator
from model import Discriminator



# from models import Generator

matplotlib.use('agg')
import matplotlib.pyplot as plt

def parse_args():
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
    parser.add_argument('--data', type=str, default='TAGAN_data',
                        help='folder of data to use')
    parser.add_argument('--del-discrim-graph', action='store_true',
                        help='whether or not to delete the discriminator\'s autograd data')
    parser.add_argument('--embedding-file', type=str, default='caption_embedding_all.pkl',
                       help='name of caption embedding file')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train for')
    parser.add_argument('--img-files', type=str, default='images',
                        help='name of image folder in data')
    parser.add_argument('--img-rep-dim', type=int, default=512,
                        help='size of the image representation')
    parser.add_argument('--lr', type=float, default=2e-3, metavar='LR',
                        help='learning rate')
    parser.add_argument('--lambda-1', type=float, default=10, metavar='L1',
                        help='weighting of conditional loss for discriminator')
    parser.add_argument('--lambda-2', type=float, default=2, metavar='L2',
                        help='weighting of reconstructive loss for generator')
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
        # 'caption_files': os.path.join(args.data, args.caption_files),
        # 'classes_file': os.path.join(args.data, args.classes_file),
        'caption_files': args.data + '/' + args.caption_files,
        'classes_file': args.data + '/' + args.classes_file,
        'conditional_weight': args.lambda_1,
        'cuda': args.cuda,
        'data': args.data,
        'date:': time.strftime("%Y-%m-%d %H:%M"),
        'del_discrim_graph': args.del_discrim_graph,
        'embedding_file': args.embedding_file, 
        'epochs': args.epochs,
        'lr': args.lr,
        'img_files': args.data + '/' + args.img_files,
        'img_rep_dim': args.img_rep_dim,
        'model_id': model_id,
        'momentum': args.momentum,
        'reconstructive_weight': args.lambda_2,
        'seed': seed,
        'schedule_epochs': args.schedule_epochs,
        'schedule_gamma': args.schedule_gamma,
        'square_momentum': args.square_momentum,
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
    if args.cuda:
        kwargs['device'] = torch.device('cuda')
    else:
        kwargs['device'] = torch.device('cpu')

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

def train(G, D, epoch, loader, txt_loader, G_optim, D_optim, val=False):
    """ Train (or validate) models for a single epoch.
    """
    val = D_optim == None or G_optim == None
    # train_loader.init_epoch()
    # pbar = tqdm(total=len(train_loader))
    # Sets model in training mode
    G.train()
    D.train()
    total_loss_G = 0
    total_loss_D = 0
    pbar = tqdm(total=len(train_loader))
    for batch_idx, (batch, txt_batch) in enumerate(zip(loader, txt_loader)):
        if not val:
            G_optim.zero_grad()
            D_optim.zero_grad()

        img = batch[0].to(kwargs['device'])
        unconditional_logits_real = D(img)
        unconditional_loss_real = F.binary_cross_entropy_with_logits(unconditional_logits_real,\
                                     torch.ones_like(unconditional_logits_real))
        

        text = batch[2].to(kwargs['device'])
        conditional_logits_real = D(img, text)
        conditional_loss_real = F.binary_cross_entropy(conditional_logits_real,\
                                     torch.ones_like(conditional_logits_real))

        text_mismatch = txt_batch[2].to(kwargs['device'])
        conditional_logits_mismatch = D(img, text_mismatch)
        conditional_loss_mismatch = F.binary_cross_entropy(conditional_logits_mismatch,\
                                     torch.zeros_like(conditional_logits_mismatch))

        fake = G(img, text_mismatch)
        unconditional_logits_fake = D(fake)
        unconditional_loss_fake = F.binary_cross_entropy_with_logits(unconditional_logits_fake,\
                                     torch.zeros_like(unconditional_logits_fake))


        #Counter for naming the images
        number = 0
        #Outputs for Last epoch
        #Better way to do this but
        # this was mainly just testing how to convert tensor
        # Can be used as a model.
        if(epoch == args.epochs - 1 or epoch == args.epochs - 2 or epoch % 10 == 0) and batch_idx == 0:
            for image_real, image_fake, txt_new in zip(img, fake, txt_batch[1]):
                number += 1
                image_real = image_real.detach().cpu().numpy()
                image_fake = image_fake.detach().cpu().numpy()
                image_real = np.transpose(image_real, (1,2,0))
                image_fake = np.transpose(image_fake, (1,2,0))
                # print("This is the image type", image.dtype)
                theString = str(number)
                drivePath = 'drive/My Drive/taganData/'
                plt.imshow(image_real)
                plt.title("Real " + theString)
                plt.savefig(drivePath + "real_" + str(epoch) + "-"+ theString + ".png")
                plt.imshow(image_fake)
                plt.title("Fake " + theString)
                plt.savefig(drivePath +"fake_" + str(epoch) + "-" + theString + ".png")
                """
                Might get error saying:
                Clipping input data to the valid range
                for imshow with RGB data ([0..1] for floats or [0..255] for integers).
                """
                #plt.imshow((image).astype('uint8'))
                with open(drivePath + 'text_' + str(epoch) + "-"+ theString + '.txt', 'w') as txt_file:
                    txt_file.write(txt_new)
            #print("-----------------------")
        elif batch_idx == 0:
            image_real = img[0]
            image_fake = fake[0]
            txt_new = txt_batch[1][0]
            image_real = image_real.detach().cpu().numpy()
            image_fake = image_fake.detach().cpu().numpy()
            image_real = np.transpose(image_real, (1,2,0))
            image_fake = np.transpose(image_fake, (1,2,0))
            # print("This is the image type", image.dtype)
            theString = str(number)
            drivePath = 'drive/My Drive/taganData/'
            plt.imshow(image_real)
            plt.title("Real " + theString)
            plt.savefig(drivePath + "real_" + str(epoch) + "-"+ theString + ".png")
            plt.imshow(image_fake)
            plt.title("Fake " + theString)
            plt.savefig(drivePath +"fake_" + str(epoch) + "-" + theString + ".png")
            """
            Might get error saying:
            Clipping input data to the valid range
            for imshow with RGB data ([0..1] for floats or [0..255] for integers).
            """
            #plt.imshow((image).astype('uint8'))
            with open(drivePath + 'text_' + str(epoch) + "-"+ theString + '.txt', 'w') as txt_file:
                txt_file.write(txt_new)

        # print('ur:', unconditional_loss_real)
        # print('uf:', unconditional_loss_fake)
        # print('cr:', conditional_loss_real)
        # print('cr logits:', conditional_logits_real)
        # print('cm:', conditional_loss_mismatch)
        # print('cm logits:', conditional_logits_mismatch)
        loss_D = unconditional_loss_real + unconditional_loss_fake + \
                kwargs['conditional_weight']*(conditional_loss_real + conditional_loss_mismatch)
        if not val:
            loss_D.backward(retain_graph=(not args.del_discrim_graph))
            D_optim.step()

        #Maybe mix up mismatching text at some point during the training?
        if args.del_discrim_graph:
            fake = G(img, text_mismatch)
            unconditional_logits_fake = D(fake)
        ### Get generator's loss
        unconditional_loss_fake = F.binary_cross_entropy_with_logits(unconditional_logits_fake,\
                                     torch.ones_like(unconditional_logits_fake))
        conditional_logits_fake = D(fake, text_mismatch)
        conditional_loss_fake = F.binary_cross_entropy(conditional_logits_fake,\
                                     torch.ones_like(conditional_logits_fake))

        # Measures dissimilarity between decoded image and input
        l2_fn = nn.MSELoss()
        reconstructive_loss = l2_fn(fake, img)
        #loss_G = kwargs['reconstructive_weight']*reconstructive_loss
        #loss_G = kwargs['reconstructive_weight']*reconstructive_loss +\
                  #unconditional_loss_fake
        loss_G = unconditional_loss_fake +\
                kwargs['conditional_weight']*conditional_loss_fake +\
                kwargs['reconstructive_weight']*reconstructive_loss
        if not val:
            loss_G.backward()
            G_optim.step()
        total_loss_G += loss_G
        total_loss_D += loss_D
        
        # Update progress
        if batch_idx % args.log_interval == 0:
            if val:
                type = 'Val'
            else:
                type = 'Train'
            avg_loss_G = total_loss_G/((batch_idx+1)*img.shape[0])
            avg_loss_D = total_loss_D/((batch_idx+1)*img.shape[0])
            print()
            print(type + ' epoch {} [{}/{} ({:.0f}%)]\tGen Loss: {:.6f}\tDisc Loss: {:.6f}'.format(
                epoch, batch_idx * img.shape[0], len(train_loader.dataset),
                100. * batch_idx / len(train_loader), avg_loss_G, avg_loss_D))
        pbar.update()
        
    print('====> Epoch: {} Average gen loss: {:.4f}\t Average disc loss: {:.4f}'.format(epoch, avg_loss_G, avg_loss_D))
    pbar.close()

    #Saving the model at various epochs.
    drivePath = 'drive/My Drive/taganData/checkpoints/'
    checkpoint = {
        'epoch': epoch,
        'genTrain': G.state_dict(),
        'discrimTrain': D.state_dict(),
        'genOptim': G_optim.state_dict(),
        'disOptim': D_optim.state_dict()
        }
    path = drivePath + 'model_' + str(epoch - epoch%10) +'.pt'
    torch.save(checkpoint, path)

        
    return avg_loss_G, avg_loss_D

if __name__ == "__main__":
    args = parse_args()
    if args.cuda:
        print('Using GPU: ' + torch.cuda.get_device_name(0))

    
    useModel = False # make true to load a model
    #make improvements later
    if useModel == True:
        drivePath = 'drive/My Drive/taganData/checkpoints/'
        newPath = drivePath + 'model_20.pt'
        checkpoint = torch.load(newPath)
        seed = set_seeds(args.seed)
        model_id, model_dir = make_model_dir(args.out_dir)
        kwargs = make_kwargs(args, seed, model_id)
        train_data = ImgCaptionData(**kwargs)
        train_loader = data.DataLoader(train_data,
                                    batch_size=args.bsize,
                                    shuffle=True)

        #txt_data = ImgCaptionData(**kwargs)
        txt_loader = data.DataLoader(train_data,
                                    batch_size=args.bsize,
                                    shuffle=True)
        # train_loader = [(0, 0)]

        # val_data = ???
        # val_loader = data.DataLoader(val_data,
                                        # batch_size=args.batch_size,
                                        # shuffle=True)
        val_loader = [(0, 0)]
        # one row of losses for training, one for testing
        losses = np.zeros((args.epochs, 4))
        G = Generator(**kwargs)
        D = Discriminator(**kwargs)
        G.load_state_dict(checkpoint['genTrain'])
        D.load_state_dict(checkpoint['discrimTrain'])
        checkpoint_epoch = checkpoint['epoch']
        optim_G = optim.Adam(G.parameters(),
                                lr=0.002,
                                betas=[args.momentum, args.square_momentum])
        optim_G.load_state_dict(checkpoint['genOptim'])
        optim_D = optim.Adam(D.parameters(),
                                lr=0.002,
                                betas=[args.momentum, args.square_momentum])
        optim_D.load_state_dict(checkpoint['disOptim'])

        # TODO maybe only get parameters in G that require gradients?
        # optim_G = optim.Adam(G.parameters(), lr=args.lr, weight_decay=1e-4)
        for epoch in range(checkpoint_epoch, args.epochs):
            # train generator
            if epoch == checkpoint_epoch:
                avg_train_loss = train(G, D, epoch, train_loader, txt_loader, optim_G, optim_D)
                losses[epoch][0] = avg_train_loss[0]
                losses[epoch][1] = avg_train_loss[1]
        
            else:
                optim_G = optim.Adam(G.parameters(),
                                    lr=0.002,
                                    betas=[args.momentum, args.square_momentum])
                optim_D = optim.Adam(D.parameters(),
                                    lr=0.002,
                                    betas=[args.momentum, args.square_momentum])
                avg_train_loss = train(G, D, epoch, train_loader, txt_loader, optim_G, optim_D)
                losses[epoch][0] = avg_train_loss[0]
                losses[epoch][1] = avg_train_loss[1]

            # test generator
            # avg_test_loss = train(G, epoch, train_loader, None)
            # losses[1][epoch] = avg_test_loss
        dest = os.path.join(model_dir, 'loss.png')
        plot_losses(losses, dest)
        
    else:    
        seed = set_seeds(args.seed)
        model_id, model_dir = make_model_dir(args.out_dir)
        kwargs = make_kwargs(args, seed, model_id)
        train_data = ImgCaptionData(**kwargs)
        train_loader = data.DataLoader(train_data,
                                    batch_size=args.bsize,
                                    shuffle=True)

        #txt_data = ImgCaptionData(**kwargs)
        txt_loader = data.DataLoader(train_data,
                                    batch_size=args.bsize,
                                    shuffle=True)
        # train_loader = [(0, 0)]

        # val_data = ???
        # val_loader = data.DataLoader(val_data,
                                        # batch_size=args.batch_size,
                                        # shuffle=True)
        val_loader = [(0, 0)]
        # one row of losses for training, one for testing
        losses = np.zeros((args.epochs, 4))
        G = Generator(**kwargs)
        D = Discriminator(**kwargs)
        # TODO maybe only get parameters in G that require gradients?
        # optim_G = optim.Adam(G.parameters(), lr=args.lr, weight_decay=1e-4)
        for epoch in range(args.epochs):
            # train generator
            optim_G = optim.Adam(G.parameters(),
                                lr=0.002,
                                betas=[args.momentum, args.square_momentum])
            optim_D = optim.Adam(D.parameters(),
                                lr=0.002,
                                betas=[args.momentum, args.square_momentum])
            avg_train_loss = train(G, D, epoch, train_loader, txt_loader, optim_G, optim_D)
            losses[epoch][0] = avg_train_loss[0]
            losses[epoch][1] = avg_train_loss[1]

            # test generator
            # avg_test_loss = train(G, epoch, train_loader, None)
            # losses[1][epoch] = avg_test_loss
        dest = os.path.join(model_dir, 'loss.png')
        plot_losses(losses, dest)
