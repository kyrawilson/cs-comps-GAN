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
        #self.img_rep_channels = kwargs["img_rep_dim"]
        modules = []
        # "num_resid_block_layers" not in kwargs
        #num_layers = kwargs["num_resid_block_layers"]
        num_layers = 3
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
        super().__init__()
        # TODO: needs to fix this kwargs as well
        #self.img_rep_channels = 512
        self.img_rep_channels = kwargs["img_rep_dim"]
        self.text_embed_size = kwargs["text_rep_dim"]


        #TEXT ENCODER
        #input: # of words in description x 300 (number of features in Fasttext embedding)
        #output: caption representation of size 256

        self.textEncoder = nn.Sequential(
        nn.GRU(300, 256, bias = False, bidirectional = True),
        nn.AvgPool1d(512, 1),
        nn.Linear(512, 256, bias = False),
        nn.LeakyReLU(0.2)
        )

        #CONDITIONING AUGMENTATION
        #input: text representation of len 256
        #output: text representation of len 128
        self.mean = nn.Sequential(
            nn.Linear(256, 128, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.log_sigma = nn.Sequential(
            nn.Linear(256, 128, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )


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
        nn.Conv2d(self.img_rep_channels+self.text_embed_size, self.img_rep_channels, 3, padding=1, bias=False),
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
        nn.Conv2d(self.img_rep_channels, int(self.img_rep_channels/2), 3, padding=1, bias=False),
        nn.BatchNorm2d(int(self.img_rep_channels/2)),
        nn.ReLU(inplace=True),
        nn.Upsample(scale_factor=2),
        nn.Conv2d(int(self.img_rep_channels/2), int(self.img_rep_channels/4), 3, padding=1, bias=False),
        nn.BatchNorm2d(int(self.img_rep_channels/4)),
        nn.ReLU(inplace=True),
        nn.Upsample(scale_factor=2),
        nn.Conv2d(int(self.img_rep_channels/4), int(self.img_rep_channels/8), 3, padding=1, bias=False),
        nn.BatchNorm2d(int(self.img_rep_channels/8)),
        nn.ReLU(inplace=True),
        nn.Conv2d(int(self.img_rep_channels/8), 3, 3, padding=1),
        nn.Tanh()
        )

        # Sends the module to CUDA if applicable
        self.to(kwargs['device'])


    #if this is getting params from __getitem__, then it should be img, description, embedding
    #may not actually need raw description at this point though
    def forward(self, img, txt):
        # image encoder
        img_feat = self.encoder(img)

        #text encoder
        txt_feat = self.textEncoder(txt)

        #conditioning augementation of data
        #Create a Gaussian distribution of text features
        z_mean = self.mean(txt_feat)
        z_log_stddev = self.log_sigma(txt_feat)
        z = torch.randn(txt_feat.size(0), 128)
        #if next(self.parameters()).is_cuda:
         #   z = z.cuda()
        txt_feat = z_mean + z_log_stddev.exp() * z

        # assume output size of text encoder is 128

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
        super().__init__()
        #Batch size from kwargs
        self.batch_size = kwargs["bsize"]
        self.image_encoder = self.ImageEncoder()

        #Calls the Unconditional discrimantor
        # Input is an image with no text
        # XXX the fourth argument in the conv2d instantiation (namely, stride) should be 0
        # BUT torch doesn't let you have 0 stride! the authors are full of it :P
        # Also, in the supplementary materials, they describe the sigmoid as a softmax, but that doesn't seem to make sense

        self.unconditional = nn.Sequential(
            nn.Conv2d(512, 1, 4, 1, padding=0, bias=False),
            nn.Sigmoid()
            )

        #Text encoder for the discriminator.
        self.textEncoderGRU = nn.Sequential(
            nn.GRU(300, 256, bias = False, bidirectional = True, batch_first=True)
        )
        self.beta_ij = nn.Sequential(
            nn.Linear(512, 3),
            nn.Softmax(dim=None)
        )
        # output size=1
        self.alpha = nn.Softmax(dim=1)
        self.weight = nn.Linear(512, 1, bias=False)
        self.bias = nn.Linear(512, 1, bias=True)
        self.local_dis = nn.Sigmoid()

        self.text_encoder = self.textEncoder()

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

        #forward function for ImageEncoder
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
                return self.gap1(img)
            img = self.conv4(img)
            if gap_layer == 4:
                return self.gap2(img)
            img = self.conv5(img)
            if gap_layer == 5:
                return self.gap3(img)
            # gap_layer is -1, so return the overall unGAP'ed result
            return img


    class Conditional(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()

        #forward function for Conditional
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

    class textEncoder(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()

            # what is the dimension for softmax function
            self.beta_ij = nn.Sequential(
                nn.Linear(512, 3),
                nn.Softmax(dim=None)
            )
            #output size=1
            self.alpha = nn.Softmax(dim=1)
            self.weight = nn.Linear(512, 1, bias=False)
            self.bias = nn.Linear(512, 1, bias=True)
            self.local_dis = nn.Sigmoid()

            
            self.textEncoderGRU = nn.Sequential(
                    nn.GRU(300, 256, bias = False, bidirectional = True, batch_first=True)
                    )
        #forward function for textEncoder
        #Can't remember commenting guidelines so it's just going here and we can change later
        #Params: txt-# words x 300, img from conv3, conv4, or conv5
        #Returns: Tensor with "score" for each word in sentence of whether or not it appears in image
        #Will need to be called after each conv layer, so join individual local_discriminator return tensors at the very end
        def forward(self,txt, img):
            
            print("I am in the forward function of text encoder")

            local_discriminator = torch.zeros(list(txt.size())[0])
            
            # Lando's change
            #Created a new GRU text encoder in this class because there was no other way to call it from discriminator.
            txt, _ = self.textEncoderGRU(txt)
            count = 0
            
            for w_i in txt:
                _weight = self.weight(w_i)
                weight = self.weight.layer.weight.view(-1,1)
                _bias = self.bias(w_i)
                bias = self.bias.layer.bias
                img = img.view(1, len(weight))
                _local_discriminator = self.local_dis(torch.mm(weight, img) + bias)
                local_discriminator[count] = _local_discriminator
                count += 1

            return local_discriminator

    #forward function for Discriminator
    def forward(self, img, txt=None):
        ''' Image encoder

        text encoder(batch_size, num_words, embedding_size)
        '''
        batch_size = len(img)
        # Unconditional discriminator
        if txt is None:
            # print(img_feats[-1][0].unsqueeze(0).size())
            #unconditional wants a 4-dimensional weight 1 512 4 4, which means it wants only one image instead of a b
            #batch of 64
            # Get unGAP'ed image encoding (thus using -1 as GAP layer)
            img_feat = self.image_encoder(-1, img)
            return self.unconditional(img_feat).squeeze()

        # Conditional discriminator

        # Throw away the second output of the GRU - it's just stuff from the last elt of the sequence
        txt_representation, _ = self.textEncoderGRU(txt)

        #alphas
        #txt_representation will relate to text encoder

        # tmp_average: (batch_size, txt_representation)
        # Our 'u' in equation 3
        tmp_average = torch.mean(txt_representation, 1)
        #adds extra dimension
        # (batch_size, 1, txt_representation)

        #batch_size, 1, text_repesentation

        represenation_size = txt_representation.shape[2]
        # Have to call contiguous() because torch is weird :P
        txt_representation = txt_representation.contiguous().view(-1, represenation_size)
        txt_representation = txt_representation.unsqueeze(1)
        tmp_average = tmp_average.view(-1,1).repeat(1, int(txt_representation.shape[0] / tmp_average.shape[0])).view(-1,tmp_average.shape[1])
        # tmp_average = tmp_average.expand(txt_representation.shape[0], tmp_average.shape[1])
        # Now I fix this by repeating the tensor until the target size
        # TODO: may need to double check if we want it this way
        tmp_average= tmp_average.unsqueeze(2)

        dot_products = torch.bmm(txt_representation, tmp_average)

        alphas = torch.zeros(batch_size, len(txt_representation))
        for i in range(batch_size):
            alpha = self.alpha(dot_products).squeeze()
            alphas[i] = alpha

        betas = torch.zeros(batch_size, len(txt_representation), 3)
        local_results = torch.zeros(batch_size, len(txt_representation), 3)

        count = 0
        for i in range(batch_size):
            for w_i in txt_representation:
                beta = self.beta_ij(w_i)
                betas[i][count] = beta
            count+= 1

        # Get GAP'ed image encodings for conditional discriminator (local discriminators)
        img_feats = []
        for gap_layer in [3, 4, 5]:
            img_feat = self.image_encoder(gap_layer, img)
            img_feats.append(img_feat)

        for i in range(batch_size):
            for j in range(3):
                print("Tell me something useful")
                print(self.text_encoder(txt, img_feats[j]))
                local_result = self.text_encoder(txt, img_feats[j])
                local_results[i] = local_result

            # local_results dimensiont: bsize in kwargs, txt length, 3
        
        weight_prod = Discriminator.Conditional(alphas, betas, local_results)
        return weight_prod

