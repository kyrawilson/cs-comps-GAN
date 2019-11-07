import torch
import torch.nn as nn


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


        self.residual_block = nn.Sequential(
        nn.Conv2D(640, 512, 3, padding=1, bias=False),
        nn.BatchNorm2D(512),
        nn.ReLU(in_place=True),
        residual_block(),
        residual_block(),
        residual_block(),
        residual_block())

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

    def forward(self, img, txt):
        # image encoder
        img_feat = self.encoder(img)

        # text encoder
        # assume output size of text encoder is 128
       
        # z_mean = self.mu(txt_feat)
        # z_log_stddev = self.log_sigma(txt_feat)
        # z = torch.randn(txt_feat.size(0), 128)
        #if next(self.parameters()).is_cuda:
         #   z = z.cuda()
        # txt_feat = z_mean + z_log_stddev.exp() * Variable(z)


        # residual block
        text_feat = text_feat.unsqueeze(-1)
        text_feat = text_feat.unsqueeze(-1)
        merge = torch.cat(txt_feat, img_feat, 1)
        merge = self.residual_block(merge)


        # decoder
        decode_img = self.decoder(img_feat) # + output_from_residual_block)
        return decode_img

