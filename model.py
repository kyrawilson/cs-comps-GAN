import torch
import torch.nn as nn
import torch.optim as optim

    # def forward(self, img, txt_feat):


    #     return img_feat, text_feat


class ResidualBlock(nn.Module):

    def __init__(self):
        super(ResidualBlock,self).__init__()
        self.residual_block = nn.Sequential(
        nn.Conv2d(512, 512, 3, padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU(),
        nn.Conv2d(512, 512, 3, padding=1),
        nn.BatchNorm2d(512),
        )
    def forward(self, x):
        return x + self.encoder(x)


class Generator(nn.Module):
    def __init__(self, **kwargs):
        super(Generator, self).__init__()

        self.textEncoder = nn.Sequential(
        #300 = dimensions of pretrained text embedding model
        #256 = number of features in the hidden state?
        #On the other hand, SISGAN uses 300 for both input and output? but not bidirectional
        #Output of shape num_directions* hidden_size, and num_directions=2 bc its bidirectional, and output from
        #supplementary is size 512?
        nn.GRU(300, 256, bias = False, bidirectional = True),

        ##TODO: temporal averaging--should potentially be added to train.py?
        #optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

        nn.Linear(512, 256, bias = False),
        nn.LeakyReLU(0.2)
        )

        #conditioning augmentation
        #chech params on this?
        self.mean = nn.Sequential(
            nn.Linear(300, 128, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.log_sigma = nn.Sequential(
            nn.Linear(300, 128, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.encoder = nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 128, 4, 2, padding=1, bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 256, 4, 2, padding=1, bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 512, 4, 2, padding=1, bias=False),
        nn.BatchNorm2d(512),
        nn.ReLU(inplace=True)
        )

        self.residualBlock = nn.Sequential(
        nn.Conv2d(640, 512, 3, padding=1, bias=False),
        nn.BatchNorm2d(512),
        nn.ReLU(inplace=True),
        ResidualBlock(),
        ResidualBlock(),
        ResidualBlock(),
        ResidualBlock(),
        )

        self.decoder = nn.Sequential(
        nn.Upsample(scale_factor=2),
        nn.Conv2d(512, 256, 3, padding=1, bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),
        nn.Upsample(scale_factor=2),
        nn.Conv2d(256, 128, 3, padding=1, bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        nn.Upsample(scale_factor=2),
        nn.Conv2d(128, 64, 3, padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 3, 3, padding=1),
        nn.Tanh()
        )

        self.to(kwargs['device'])


    #if this is getting params from __getitem__, then it should be img, description, embedding
    #may not actually need raw description at this point though
    def forward(self, img, txt):
        # image encoder
        img_feat = self.encoder(img)

        #text encoder
        txt_feat = self.textEncoder(txt)

        #conditioning augementation of data
        z_mean = self.mean(txt_feat)
        z_log_stddev = self.log_sigma(txt_feat)
        z = torch.randn(txt_feat.size(0), 128)
        #if next(self.parameters()).is_cuda:
         #   z = z.cuda()
        txt_feat = z_mean + z_log_stddev.exp() * Variable(z)

        # assume output size of text encoder is 128

        # residual block



        # decoder
        decode_img = self.decoder(img_feat) # + output_from_residual_block)
        return decode_img
