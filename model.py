import torch
import torch.nn as nn
import torch.optim as optim


# def forward(self, img, txt_feat):


#     return img_feat, text_feat


class ResidualBlock(nn.Module):
    '''
    The class configures layers for individual residual block.
    The input of first residual block is concatenated image represenation and text embedding.
    The input of the remaining residual blocks is output from last residual block.
    Outputs the representation of input plus modified version of input.
    Input dimension: img_rep_channels*width*height
    Output dimension: img_rep_channels*width*height
    '''

    def __init__(self, **kwargs):
        super().__init__()
        # Needs to be fixed though I do not know why because it's in kwargs
        self.img_rep_channels = 512
        # self.img_rep_channels = kwargs["img_rep_dim"]
        modules = []
        # "num_resid_block_layers" not in kwargs
        # num_layers = kwargs["num_resid_block_layers"]
        num_layers = 3
        for i in range(num_layers - 1):
            modules.append(nn.Conv2d(self.img_rep_channels, self.img_rep_channels, 3, padding=1))
            modules.append(nn.BatchNorm2d(self.img_rep_channels))
            modules.append(nn.ReLU(inplace=True))

        modules.append(nn.Conv2d(self.img_rep_channels, self.img_rep_channels, 3, padding=1))
        modules.append(nn.BatchNorm2d(self.img_rep_channels))
        self.residual_block = nn.Sequential(*modules)

    #         self.residual_block = nn.Sequential(
    #         nn.Conv2d(512, 512, 3, padding=1),
    #         nn.BatchNorm2d(512),
    #         nn.ReLU(inplace=True),
    #         nn.Conv2d(512, 512, 3, padding=1),
    #         nn.BatchNorm2d(512),
    #         )
    def forward(self, x):
        return x + self.residual_block(x)


class Generator(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        # TODO: needs to fix this kwargs as well
        # self.img_rep_channels = 512
        self.img_rep_channels = kwargs["img_rep_dim"]
        self.text_embed_size = kwargs["text_rep_dim"]
        self.text_encoder = self.TextEncoder(**kwargs)

        # Applies 2D Convolution over an input signal
        # 4 different layers with different input and output sizes in each
        # Input size: 3 output size: 64, conv2d(3,1)
        # Input size: 64 output size: 128, conv2d(4,2)
        # Input size: 128 output size: 256, conv2d(4,2)
        # Input size: 256 output size: 512, conv2d(4,2)
        # With Batch normalization after each layer.

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

        # input size of residual block is image representation channels(512) + text embedding channels(128)
        # output size of image representaion channels
        self.modifier = nn.Sequential(
            nn.Conv2d(self.img_rep_channels + self.text_embed_size, self.img_rep_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.img_rep_channels),
            nn.ReLU(inplace=True),
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock())

        # input of output of modifier(residual blocks as a whole): 512*16*16
        # output of image size: 3*128*128
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.img_rep_channels, int(self.img_rep_channels / 2), 3, padding=1, bias=False),
            nn.BatchNorm2d(int(self.img_rep_channels / 2)),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(int(self.img_rep_channels / 2), int(self.img_rep_channels / 4), 3, padding=1, bias=False),
            nn.BatchNorm2d(int(self.img_rep_channels / 4)),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(int(self.img_rep_channels / 4), int(self.img_rep_channels / 8), 3, padding=1, bias=False),
            nn.BatchNorm2d(int(self.img_rep_channels / 8)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(self.img_rep_channels / 8), 3, 3, padding=1),
            nn.Tanh()
        )

        # Sends the module to CUDA if applicable
        self.to(kwargs['device'])

    # TEXT ENCODER
    # input: # of words in description x 300 (number of features in Fasttext embedding)
    # output: caption representation of size 256

    class TextEncoder(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.gru = nn.GRU(300, 256, bias=False, bidirectional=True, batch_first=True)
            self.collapse_bidir = nn.Sequential(
                nn.Linear(512, 256, bias=False),
                nn.LeakyReLU(0.2)
            )

            # CONDITIONING AUGMENTATION
            # input: text representation of len 256
            # output: text representation of len 128
            self.mean = nn.Sequential(
                nn.Linear(256, 128, bias=False),
                nn.LeakyReLU(0.2, inplace=True)
            )

            self.log_sigma = nn.Sequential(
                nn.Linear(256, 128, bias=False),
                nn.LeakyReLU(0.2, inplace=True)
            )

            self.to(kwargs['device'])
            self.device = kwargs['device']

        def forward(self, txt):
            '''
            txt: (bsize, n_words, txt_embed_size)
            '''
            # text encoder
            txt_feat, _ = self.gru(txt)
            # (bsize, n_words, txt_rep_size)
            # average over first dimension (n_words) for tmp average
            tmp_average = torch.mean(txt_feat, 1)
            # (bsize, txt_rep_size)
            tmp_average = self.collapse_bidir(tmp_average)
            # (bsize, txt_rep_size/2)

            # conditioning augementation of data
            # Create a Gaussian distribution of text features
            z_mean = self.mean(tmp_average)
            # (bsize, txt_rep_size/4)
            z_log_stddev = self.log_sigma(tmp_average)
            # (bsize, txt_rep_size/4)
            bsize = txt_feat.shape[0]
            z = torch.randn(bsize, 128).to(self.device)
            # (bsize, txt_rep_size/4)
            # if next(self.parameters()).is_cuda:
            #   z = z.cuda()
            txt_feat = z_mean + z_log_stddev.exp() * z
            # (bsize, txt_rep_size/4)
            return txt_feat

    # if this is getting params from __getitem__, then it should be img, description, embedding
    # may not actually need raw description at this point though
    def forward(self, img, txt):
        '''
        img: (bsize, n_channels, w, h)
        txt: (bsize, n_words, txt_embed_size)
        '''
        # image encoder
        img_feat = self.encoder(img)
        txt_feat = self.text_encoder(txt)
        # (bsize, txt_rep_size/4)

        # residual block
        # concatenate text embedding with image represenation
        txt_feat = txt_feat.unsqueeze(-1)
        txt_feat = txt_feat.unsqueeze(-1)
        txt_feat = txt_feat.expand(-1, -1, 16, 16)
        # (bsize, txt_rep_size/4, 16, 16) (where 16 is w and h of the img)
        merge = torch.cat([txt_feat, img_feat], dim=1)
        # (bsize, img_rep_size + txt_rep_size/4, 16, 16)
        merge = self.modifier(merge)
        # (bsize, img_rep_size, 16, 16)

        # decoder
        # change img_feat to merge when testing with residual blocks
        decode_img = self.decoder(img_feat)  # + output_from_residual_block)
        # (bsize, n_channels, w, h)
        return decode_img


class Discriminator(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        # Batch size from kwargs
        self.batch_size = kwargs["bsize"]
        self.image_encoder = self.ImageEncoder()

        # Calls the Unconditional discrimantor
        # Input is an image with no text
        # XXX the fourth argument in the conv2d instantiation (namely, stride) should be 0
        # BUT torch doesn't let you have 0 stride! the authors are full of it :P
        # Also, in the supplementary materials, they describe the sigmoid as a softmax, but that doesn't seem to make sense

        self.unconditional = nn.Sequential(
            nn.Conv2d(512, 1, 4, 1, padding=0, bias=False),
            nn.Sigmoid()
        )

        self.conditional = self.Conditional()

        # Text encoder for the discriminator.
        self.text_encoder_GRU = nn.Sequential(
            nn.GRU(300, 256, bias=False, bidirectional=True, batch_first=True)
        )
        self.beta = nn.Sequential(
            nn.Linear(512, 3),
            nn.Softmax(dim=2)
        )
        # output size=1
        self.alpha = nn.Softmax(dim=1)
        self.LD_weights = [nn.Linear(512, 256, bias=False),
                           nn.Linear(512, 512, bias=False),
                           nn.Linear(512, 512, bias=False)]
        self.LD_biases = [nn.Linear(512, 1, bias=True),
                          nn.Linear(512, 1, bias=True),
                          nn.Linear(512, 1, bias=True)]

        # Sends the model to CUDA if applicable
        self.to(kwargs['device'])
        self.LD_weights = [m.to(kwargs['device']) for m in self.LD_weights]
        self.LD_biases = [m.to(kwargs['device']) for m in self.LD_biases]

    class ImageEncoder(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()

            self.conv123 = nn.Sequential(
                nn.Conv2d(3, 64, 4, 2, padding=1, bias=False),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(64, 128, 4, 2, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(128, 256, 4, 2, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
            self.conv4 = nn.Sequential(
                nn.Conv2d(256, 512, 4, 2, padding=1, bias=False),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
            self.conv5 = nn.Sequential(
                nn.Conv2d(512, 512, 4, 2, padding=1, bias=False),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )

            self.gap1 = nn.Sequential(
                nn.Conv2d(256, 256, 3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.AvgPool2d(16, stride=None, padding=0)
            )
            self.gap2 = nn.Sequential(
                nn.Conv2d(512, 512, 3, padding=1, bias=False),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.AvgPool2d(8, stride=None, padding=0)
            )
            self.gap3 = nn.Sequential(
                nn.Conv2d(512, 512, 3, padding=1, bias=False),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.AvgPool2d(4, stride=None, padding=0)
            )

        # forward function for ImageEncoder
        def forward(self, gap_layer, img):
            '''
            gap_layer: int (3, 4, or 5), layer after which to output GAP
                If gap_layer is set to -1, then don't do GAP, and output normal
                result after last conv layer
            img: shape(batch size, width, height, num channels)
            '''
            assert gap_layer in [-1, 3, 4, 5], "gap_layer must be -1, 3, 4, or 5"
            img = self.conv123(img)
            if gap_layer == 3:
                return self.gap1(img).squeeze()
            img = self.conv4(img)
            if gap_layer == 4:
                return self.gap2(img).squeeze()
            img = self.conv5(img)
            if gap_layer == 5:
                return self.gap3(img).squeeze()
            # gap_layer is -1, so return the overall unGAP'ed result
            return img

    class Conditional(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()

        # forward function for Conditional
        def forward(self, alphas, betas, local_results):
            '''
            alphas: (batch_size, num_words)
            betas: (batch_size, num_words, 3)
            local_results: (batch_size, num_words, 3)
            '''

            # Add together last dimension of local results with betas
            n_words = local_results.shape[1]
            local_results_long = local_results.view(-1, 3)
            # (bsize*n_words, 3)
            local_results_long = local_results_long.unsqueeze(1)
            # (bsize*n_words, 1, 3)
            betas_long = betas.view(-1, 3)
            # (bsize*n_words, 3)
            betas_long = betas_long.unsqueeze(2)
            # (bsize*n_words, 3, 1)

            # (bsize*n_words, 1, 3) @ (bsize*n_words, 3, 1)
            weighted_sum = torch.bmm(local_results_long, betas_long).squeeze()
            # (bsize*n_words)
            weighted_sum = weighted_sum.view(-1, n_words)
            # (bsize, n_words)
            # Multiply together second dimension of local results with alphas
            # First raise weighted sums to alphas
            weighted_prod = torch.pow(weighted_sum, alphas)
            weighted_prod = torch.prod(weighted_prod, dim=1)
            # (bsize)
            return weighted_prod

    # forward function for Discriminator
    def forward(self, img, txt=None):
        ''' Image encoder

        text encoder(batch_size, num_words, embedding_size)
        '''
        batch_size = len(img)
        # Unconditional discriminator
        if txt is None:
            # print(img_feats[-1][0].unsqueeze(0).size())
            # unconditional wants a 4-dimensional weight 1 512 4 4, which means it wants only one image instead of a b
            # batch of 64
            # Get unGAP'ed image encoding (thus using -1 as GAP layer)
            img_feat = self.image_encoder(-1, img)
            return self.unconditional(img_feat).squeeze()

        # Conditional discriminator

        # Throw away the second output of the GRU - it's just stuff from the last elt of the sequence
        txt_rep, _ = self.text_encoder_GRU(txt)

        # alphas
        # txt_rep will relate to text encoder

        # tmp_average: (batch_size, txt_rep)
        # Our 'u' in equation 3
        tmp_average = torch.mean(txt_rep, 1)
        # (bsize, txt_rep_size)

        rep_size = txt_rep.shape[2]
        # Have to call contiguous() because torch is weird :P
        txt_rep_flat = txt_rep.contiguous().view(-1, rep_size)
        # (bsize*n_words, rep_size)
        txt_rep_flat = txt_rep_flat.unsqueeze(1)
        # (bsize*n_words, 1, txt_rep_size)
        n_words = txt_rep.shape[1]
        tmp_average = tmp_average.repeat(n_words, 1)
        # (bsize*n_words, txt_rep_size)
        tmp_average = tmp_average.unsqueeze(2)
        # (bsize*n_words, txt_rep_size, 1)

        # (bsize*n_words, 1, txt_rep_size) @ (bsize*n_words, txt_rep_size, 1)
        dot_products = torch.bmm(txt_rep_flat, tmp_average)
        dot_products = dot_products.view(-1, n_words)
        # (bsize, n_words)
        alphas = self.alpha(dot_products)
        # txt_rep: (bsize, n_words, txt_rep_size)
        betas = self.beta(txt_rep)
        # betas: (bsize, n_words, 3)

        # Get GAP'ed image encodings for conditional discriminator (local discriminators)
        img_feats = []
        for gap_layer in [3, 4, 5]:
            img_feat = self.image_encoder(gap_layer, img)
            img_feats.append(img_feat)

        LD_results_by_conv = []
        for j in range(3):
            # forward function for TextEncoder
            # Can't remember commenting guidelines so it's just going here and we can change later
            # Params: txt-# words x 300, img from conv3, conv4, or conv5
            # Returns: Tensor with "score" for each word in sentence of whether or not it appears in image
            # Will need to be called after each conv layer, so join individual local_discriminator return tensors at the very end
            ''' 
            Params: txt-# words x 300;
                img_rep from conv3, conv4, or conv5: (bsize, 256) or (bsize, 512), respectively

            Returns: Tensor with "score" for each word in sentence of whether or not it appears in image
            Will need to be called after each conv layer, so join individual local_discriminator return tensors at the very end
            '''

            # Lando's change
            # Created a new GRU text encoder in this class because there was no other way to call it from discriminator.
            img_rep = img_feats[j]
            # (bsize, img_rep_size)
            img_rep_size = img_feats[j].shape[1]

            weight_net = self.LD_weights[j]
            # txt_rep: (bsize, n_words, txt_rep_size)
            _weight = weight_net(txt_rep)
            # (bsize, n_words, img_rep_size)
            weight = _weight.view(-1, img_rep_size)
            # (bsize*n_words, img_rep_size)
            weight = weight.unsqueeze(1)
            # (bsize*n_words, 1, img_rep_size)
            bias_net = self.LD_biases[j]
            bias = bias_net(txt_rep).squeeze()
            # (bsize, n_words)
            # Repeat img_rep n_words times
            img_rep = img_rep.repeat(_weight.shape[1], 1)
            # (bsize*n_words, img_rep_size)
            img_rep = img_rep.unsqueeze(2)
            # (bsize*n_words, img_rep_size, 1)

            # (bsize*n_words, 1, img_rep_size) @ (bsize*n_words, img_rep_size, 1)
            dot_prods = torch.bmm(weight, img_rep).squeeze()
            # (bsize*n_words)
            n_words = txt_rep.shape[1]
            dot_prods = dot_prods.view(-1, n_words)
            # (bsize, n_words)
            dot_prods += bias
            LD_result = nn.Sigmoid()(dot_prods)
            # Unsqueeze last dim to concatenate later
            LD_results_by_conv.append(LD_result.unsqueeze(2))

        # local_results dimension: (bsize in kwargs, txt length, 3)

        local_results = torch.cat(LD_results_by_conv, dim=2)
        weight_prod = self.conditional(alphas, betas, local_results)
        return weight_prod

