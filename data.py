'''
PARAMETERS AND CALL (for testing purposes)

FT_file = "mini_TAGAN_data/cc.en.300.bin"
img_files = "mini_TAGAN_data/images"
caption_files = "mini_TAGAN_data/text_c10"
classes_file = "mini_TAGAN_data/classes.txt"
img_transform = transforms.Compose([transforms.Resize((136,136)),
                                         transforms.RandomCrop(128),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.RandomRotation(10),
                                         transforms.ToTensor()])

self.data = self.load_dataset(img_files, caption_files, classes_file)

test = ImgCaptionData(img_files, caption_files, classes_file, img_transform = None)
test = ImgCaptionData(**kwargs)
'''

import torch
import fasttext
import os
import random

import torch.utils.data as data
import torchvision.transforms as transforms

import pickle

from string import digits
from PIL import Image

FT_file = "mini_TAGAN_data/cc.en.300.bin"
img_files = "mini_TAGAN_data/images"
caption_files = "mini_TAGAN_data/text_c10"
classes_file = "mini_TAGAN_data/classes.txt"
img_transform = transforms.Compose([transforms.Resize((136,136)),
                                         transforms.RandomCrop(128),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.RandomRotation(10),
                                         transforms.ToTensor()])

class ImgCaptionData(data.Dataset):

    def __init__(self, **kwargs):
        #super(whatever)__init__()?
        #print("Loading fasttext model...")
        #self.word_embedding = fasttext.load_model(FT_file)
        #print("Fast text is loaded!")

        self.word_embedding = pickle.load(open( "caption_embedding.pkl", "rb" ))
        self.data = self.load_dataset(kwargs['img_files'], kwargs['caption_files'], kwargs['classes_file'])

        #add in kwargs here
        self.img_transform = transforms.Compose([transforms.Resize((136,136)),
                                                 transforms.RandomCrop(128),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.RandomRotation(10),
                                                 transforms.ToTensor()])

        self.max_word_length = 50

        if kwargs['img_transform'] == None:
            img_transform = transforms.ToTensor()


    #Load images and captions into list of dicts, also add word embedding
    def load_dataset(self, img_files, caption_files, classes_file):
        output = []
        with open(classes_file) as f:
            classes = f.readlines()
            for class_name in classes:

                #This part is just to edit caption file from CUB, if we make our own it may not be needed
                class_name = class_name.rstrip("\n")
                class_name = class_name.lstrip(digits)
                class_name = class_name.lstrip(" ")

                captions = os.listdir(os.path.join(caption_files,class_name))
                for caption in captions:
                    image_path = os.path.join(img_files, class_name, caption.replace("txt", "jpg"))
                    caption_path = os.path.join(caption_files, class_name, caption)

                    if not(caption_path.startswith("._")):
                        with open(caption_path) as f2:
                            caption_list = f2.readlines()
                            #Might need to strip newline char here
                            output.append({
                                'img': image_path,
                                'caption': caption_list,
                                'embedding': self.get_word_embedding_fast(caption_path),
                                'class_name': class_name
                                })
                            f2.close()
        f.close()
        return output

    def get_word_embedding(self, caption_list):
        #Need to have the length of the description for something?->add later when necessary
        #do we want single tensor for entire sentence or list of tensors for each word?
        output = []
        for caption in caption_list:
            temp_caption = caption.split()
            temp_caption[len(temp_caption)-1] = temp_caption[len(temp_caption)-1].rstrip(".")
            #Tensor of list of word vectors
            word_vecs = torch.Tensor([self.word_embedding[w.lower()] for w in temp_caption])
            #Don't hard code in 50 here, supposed to be a reference to max_word_length
            if len(temp_caption) < 50:
                    word_vecs = torch.cat((
                    word_vecs,
                    torch.zeros(50 - len(temp_caption), word_vecs.size(1))
                ))
            #Add tensor representing one caption to list of all caption tensors
            output.append(word_vecs)

        #This line was in the original code, but I'm not totally sure what the point is...
        #output = torch.stack(output)

        return output

    def get_word_embedding_fast(self, caption_path):
        return self.word_embedding[caption_path]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        #probably don't need to return raw description
        value = self.data[index]
        image = Image.open(value['img'])
        image = self.img_transform(image)
        randIndex = random.randint(0,len(value['caption'])-1)
        description = value['caption'][randIndex]
        embedding = value['embedding'][randIndex]
        class_name = value['class_name'][randIndex]
        return image, description, embedding, class_name

#test = ImgCaptionData(img_files, caption_files, classes_file, img_transform = None)

###TODO:
#right now classes file is all of the classes
###
