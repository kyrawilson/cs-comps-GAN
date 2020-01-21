import torch
import torch.nn as nn


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
        super(ResidualBlock,self).__init__()
        self.img_rep_channels = kwargs["img_rep_channels"]
        modules = []
        num_layers = kwargs["num_resid_block_layers"]
        for i in range(num_layers-1):
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
        super(Generator, self).__init__()
        self.img_rep_channels = kwargs["img_rep_channels"]
        self.text_embed_size = kwargs["text_embed_size"]

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
        nn.Conv2D(self.img_rep_channels+self.text_embed_size, self.img_rep_channels, 3, padding=1, bias=False),
        nn.BatchNorm2D(self.img_rep_channels),
        nn.ReLU(in_place=True),
        residual_block(),
        residual_block(),
        residual_block(),
        residual_block())

        # input of output of modifier(residual blocks as a whole): 512*16*16
        # output of image size: 3*128*128
        self.decoder = nn.Sequential(
        nn.Upsample(scale_factor=2),
        nn.Conv2d(self.img_rep_channels, self.img_rep_channels/2, 3, padding=1, bias=False),
        nn.BatchNorm2d(self.img_rep_channels/2),
        nn.ReLU(inplace=True),
        nn.Upsample(scale_factor=2),
        nn.Conv2d(self.img_rep_channels/2, self.img_rep_channels/4, 3, padding=1, bias=False),
        nn.BatchNorm2d(self.img_rep_channels/4),
        nn.ReLU(inplace=True),
        nn.Upsample(scale_factor=2),
        nn.Conv2d(self.img_rep_channels/4, self.img_rep_channels/8, 3, padding=1, bias=False),
        nn.BatchNorm2d(self.img_rep_channels/8),
        nn.ReLU(inplace=True),
        nn.Conv2d(self.img_rep_channels/8, 3, 3, padding=1),
        nn.Tanh()
        )

        # Sends the module to CUDA if applicable
        self.to(kwargs['device'])

    def forward(self, img, txt):
        # image encoder
        img_feat = self.encoder(img)

        # text encoder

        # residual block
        # concatenate text embedding with image represenation
        text_feat = text_feat.unsqueeze(-1)
        text_feat = text_feat.unsqueeze(-1)
        merge = torch.cat(txt_feat, img_feat, 1)

        merge = self.modifier(merge)


        # decoder
        # change img_feat to merge when testing with residual blocks
        decode_img = self.decoder(img_feat) # + output_from_residual_block)
        return decode_img

class Discriminator(nn.Module):
    def __init__(self, **kwargs):
        super(Discriminator, self).__init__()

        class ImageEncoder(nn.Module):
            def __init__(self, **kwargs):
                super(ImageEncoder, self).__init__()

            self.conv123 = nn.Sequential(
                nn.Conv2d(3, 64, 4, 2, padding=1, bias=False),
                nn.LeakyReLU(negative_slope=0.2, inplace=False),
                nn.Conv2d(64, 128, 4, 2, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(negative_slope=0.2, inplace=False),
                n.Conv2d(128, 256, 4, 2, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(negative_slope=0.2, inplace=False)
            )
            self.conv4 = nn.Sequential(
                n.Conv2d(256, 512, 4, 2, padding=1, bias=False),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(negative_slope=0.2, inplace=False)
            )
            self.conv5 = nn.Sequential(
                n.Conv2d(512, 512, 4, 2, padding=1, bias=False),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(negative_slope=0.2, inplace=False)
            )

            def forward(self, gap_layer, img):
            '''
            gap_layer: int (3, 4, or 5), layer after which to output GAP
            img: shape(batch size, width, height, num channels)
            '''
                assert gap_layer in range(3,6), "gap_layer must be 3, 4, or 5"
                img = self.conv123(img)
                if gap_layer == 3:
                    return nn.AvgPool2d(16, stride=None, padding=0).forward(img)
                img = self.conv4(img)
                if gap_layer == 4:
                    return nn.AvgPool2d(8, stride=None, padding=0).forward(img)
                img = self.conv5(img)
                return nn.AvgPool2d(4, stride=None, padding=0).forward(img)

        class Unconditional(nn.Module):
            def __init__(self, **kwargs):
                super(Unconditional, self).__init__()

            self.conv = nn.Sequential(
                nn.Conv2d(512, 1, 4, 0, padding=1, bias=False)
                nn.Softmax(dim=None)
                )

        class Conditional(nn.Module):
            def __init__(self, **kwargs):
                super(Conditional, self).__init__()

            
            def forward(self, alphas, betas, local_results):
                '''
                alphas: (batch_size, num_words)
                betas: (batch_size, num_words, 3)
                local_results: (batch_size, num_words, 3)
                '''

                # Add together last dimension of local results with betas
                n_words = local_results.shape[1]
                local_results_long = local_results.view(-1, 3)
                local_results_long = local_results_long.unsqueeze(2)
                betas_long = betas.view(-1, 3)
                betas_long = betas_long.unsqueeze(1)
                weighted_sum = torch.bmm(local_results_long, betas_long).squeeze()
                weighted_sum = weighted_sum.view(-1, n_words)
                # Multiply together second dimension of local results with alphas
                # First raise weighted sums to alphas
                weighted_prod = torch.pow(weighted_sum, alphas)
                weighted_prod = torch.prod(weighted_prod)
                return weighted_prod
                # In discriminator write the alpha and beta classes.


       def forward(self, img, txt): 
        ''' Image encoder

        text encoder(batch_size, num_words, embedding_size)
        '''

        #txt_representation will relate to text encoder

        # tmp_average: (batch_size, txt_representation)
        # Our 'u' in equation 3
        tmp_average = torch.mean(txt_representation, 1)
        #adds extra dimension
        # (batch_size, 1, txt_representation)
        
        #batch_size, 1, text_repesentation
        represenation_size = txt_representation.shape[2]
        txt_representation = txt_representation.view(-1, represenation_size)
        txt_representation = txt_representation.unsqueeze(1)

        tmp_average = tmp_average.expand(txt_representation.shape[0], tmp_average.shape[1])
        tmp_average= tmp_average.unsqueeze(2)

        dot_products = torch.bmm(txt_representation, tmp_average)
        #ToDo: Exponentiate

        

        















