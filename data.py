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

test = ImgCaptionData(**kwargs)                                       
'''

import torch
import fasttext
import os
import random

import torch.utils.data as data
import torchvision.transforms as transforms

from string import digits
from PIL import Image

class ImgCaptionData(data.Dataset):

    #def __init__(self, img_files, caption_files, classes_file, img_transform = None):
    def __init__(self, **kwargs):
        #super(whatever)__init__()?
        #self.word_embedding = fasttext.load_model(FT_file)
        self.data = self.load_dataset(kwargs['img_files'], kwargs['caption_files'], kwargs['classes_file'])
        self.img_transform = kwargs['img_transform']

        if kwargs['img_transform'] == None:
            img_transform = transforms.ToTensor()


    #Load images and captions into list of dicts, also add word embedding
    def load_dataset(self, img_files, caption_files, classes_file):
        output = []
        with open(classes_file) as f:
            classes = f.readlines()
            print(classes)
            for class_name in classes:

                #This part is just to edit caption file from CUB, if we make our own it may not be needed
                class_name = class_name.rstrip("\n")
                class_name = class_name.lstrip(digits)
                class_name = class_name.lstrip(" ")
                
                captions = os.listdir(os.path.join(caption_files,class_name))
                print(captions)
                for caption in captions:
                    image_path = os.path.join(img_files, class_name, caption.replace("txt", "jpg"))
                    caption_path = os.path.join(caption_files, class_name, caption)
                    with open(caption_path) as f2:
                        caption_list = f2.readlines()
                        #Might need to strip newline char here
                        #Eventually need word embeddings to be Tensors
                        output.append({'img': image_path, 'caption': caption_list, 'embedding': self.get_word_embedding(caption)})
                        f2.close()
        f.close()
        #print(output)
        return output

    #Need to write this function
    def get_word_embedding(self, caption):
        return 0

    def __len__(self, data):
        return len.data()

    def __getitem__(self, index):
        value = self.data[index]
        image = Image.open(value['img'])
        image = self.img_transform(image)
        description = random.choice(value['caption'])
        embedding = value['embedding']
        return image, description, embedding


###TODO:
#right now classes file is all of the classes
###
