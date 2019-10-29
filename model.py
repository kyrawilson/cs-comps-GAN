import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock,self).__init__()
        self.residual_block = nn.Sequential(
        nn.Conv2D(512, 512, 3, padding=1),
        nn.BatchNorm2D(512),
        nn.ReLU(),
        nn.Conv2D(512, 512, 3, padding=1),
        nn.BatchNorm2D(512),
        )
    def forward(self, x):
        return x + self.encoder(x)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.residualBlock = nn.Sequential(
        nn.Conv2D(640, 512, 3, padding=1, bias=False),
        nn.BatchNorm2D(512),
        nn.ReLU(in_place=True),
        residual_block(),
        residual_block()
        residual_block(),
        residual_block()
        )

        self.decoder = nn.Sequential(
        nn.Upsample(scale_factor=2),
        nn.Conv2D(512, 256, 3, padding=1, bias=False),
        nn.BatchNorm2D(256),
        nn.ReLU(in_place=True),
        nn.Upsample(scale_factor=2),
        nn.Conv2D(256, 128, 3, padding=1, bias=False),
        nn.BatchNorm2D(128),
        nn.ReLU(in_place=True),
        nn.Upsample(scale_factor=2),
        nn.Conv2D(128, 64, 3, padding=1, bias=False),
        nn.BatchNorm2D(64),
        nn.ReLU(in_place=True),
        nn.Conv2D(64, 3, 3, padding=1),
        nn.Tanh()
        )

    def forward(self, img, txt):
        # image encoder
        encode_img = self.encoder(img)

        # text encoder
        # assume output size of text encoder is 128

        # residual block



        # decoder
        decode_img = self.decoder(encode_img + output_from_residual_block)
