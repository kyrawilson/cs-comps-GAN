'''
PARAMETERS AND CALL

FT_file = "mini_TAGAN_data/cc.en.300.bin"
img_files = "mini_TAGAN_data/images"
caption_files = "mini_TAGAN_data/text_c10"
classes_file = "mini_TAGAN_data/classes.txt"
img_transform = transforms.Compose([transforms.Resize((136,136)),
                                         transforms.RandomCrop(128),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.RandomRotation(10),
                                         transforms.ToTensor()])

test = ImgCaptionData(img_files, caption_files, classes_file, img_transform = None)
test = ImgCaptionData(**kwargs)
'''

import torch
import fasttext
import os
import random

import torch.utils.data as data
#import torchvision.transforms as transforms

from string import digits
from PIL import Image

FT_file = "mini_TAGAN_data/cc.en.300.bin"
img_files = "mini_TAGAN_data/images"
caption_files = "mini_TAGAN_data/text_c10"
classes_file = "mini_TAGAN_data/classes.txt"

'''
img_transform = transforms.Compose([transforms.Resize((136,136)),
                                         transforms.RandomCrop(128),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.RandomRotation(10),
                                         transforms.ToTensor()])
'''

class ImgCaptionData(data.Dataset):

    def __init__(self, img_files, caption_files, classes_file, img_transform = None):
    #def __init__(self, **kwargs):
        #super(whatever)__init__()?
        print("Loading fasttext model...")
        self.word_embedding = fasttext.load_model(FT_file)
        print("Fast text is loaded!")
        #self.data = self.load_dataset(kwargs['img_files'], kwargs['caption_files'], kwargs['classes_file'])
        self.data = self.load_dataset(img_files, caption_files, classes_file)
        #add in kwargs here
        self.img_transform = img_transform

        #if kwargs['img_transform'] == None:
            #img_transform = transforms.ToTensor()


    #Load images and captions into list of dicts, also add word embedding
    def load_dataset(self, img_files, caption_files, classes_file):
        output = []
        with open(classes_file) as f:
            classes = f.readlines()
            #print(classes)
            for class_name in classes:

                #This part is just to edit caption file from CUB, if we make our own it may not be needed
                class_name = class_name.rstrip("\n")
                class_name = class_name.lstrip(digits)
                class_name = class_name.lstrip(" ")
                #print(os.path.join(caption_files,class_name))
                captions = os.listdir(os.path.join(caption_files,class_name))
                #print(captions)
                for caption in captions:
                    image_path = os.path.join(img_files, class_name, caption.replace("txt", "jpg"))
                    caption_path = os.path.join(caption_files, class_name, caption)
                    #print(caption_path)

                    if not(caption_path.startswith("._")):
                        with open(caption_path) as f2:
                            caption_list = f2.readlines()
                            #Might need to strip newline char here
                            #Eventually need word embeddings to be Tensors
                            output.append({'img': image_path, 'caption': caption_list, 'embedding': self.get_word_embedding(caption_list)})
                            f2.close()
        f.close()
        #print(output)
        return output

    #Need to write this function
    def get_word_embedding(self, caption_list):
        #Should make tensor of embeddings for each caption? --> so should end up with a list of Tensors?
        #Why zero-pad word vectors? (and is max_word_length needed?)
        #What is purpose of num2chars function? --> not totally sure it is needed bc I'm using text (instead of Tensory-thing) anyways...?
        #Make sure what function is returning is what we actually want
        output = []
        for caption in caption_list:
            temp_caption = caption.split()
            word_vecs = torch.Tensor([self.word_embedding[w.lower()] for w in temp_caption])
            output.append(word_vecs)
        print(output)
        return output

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        value = self.data[index]
        image = Image.open(value['img'])
        #image = self.img_transform(image)
        randIndex = random.randint(0,len(value['caption']));
        description = value['caption'][randIndex]
        embedding = value['embedding'][randIndex]
        return image, description, embedding

test = ImgCaptionData(img_files, caption_files, classes_file, img_transform = None)

###TODO:
#right now classes file is all of the classes
###
