import torch
import fasttext
import os
import random

import torch.utils.data as data
import torchvision.transforms as transforms

import pickle

from string import digits
from PIL import Image

folder = "mini_TAGAN_data_"
FT_file = "TAGAN_data/cc.en.300.bin"
img_files = "/images"
caption_files = "/text_c10"
classes_file = "TAGAN_data/classes.txt"
img_transform = transforms.Compose([transforms.Resize((136,136)),
                                         transforms.RandomCrop(128),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.RandomRotation(10),
                                         transforms.ToTensor()])

def load_dataset(img_files, caption_files, classes_file):
    output = {}
    with open(classes_file) as f:
        classes = f.readlines()
        i = 1
        for class_name in classes:

            print(class_name)
            #This part is just to edit caption file from CUB, if we make our own it may not be needed
            class_name = class_name.rstrip("\n")
            class_name = class_name.lstrip(digits)
            class_name = class_name.lstrip(" ")

            if i % 5 == 0:
                folder_num = i // 5
            else:
                folder_num = i // 5 + 1

            folder = "mini_TAGAN_data_"
            img_files = "/images"
            caption_files = "/text_c10"
            folder = folder + str(folder_num)
            caption_files = folder + caption_files
            img_files = folder + img_files

            captions = os.listdir(os.path.join(caption_files,class_name))
            for caption in captions:
                if not(caption.startswith("._")):
                    image_path = os.path.join(img_files, class_name, caption.replace("txt", "jpg"))
                    caption_path = os.path.join(caption_files, class_name, caption)

                    if not(caption_path.startswith("._")):
                        with open(caption_path, 'r') as f2:
                            #print(caption_path)
                            caption_list = f2.readlines()
                            #Might need to strip newline char here
                            #print(caption_list)
                            embedding = get_word_embedding(caption_list)
                            #temp_caption_path = "TAGAN_data" + caption_files + class_name + caption
                            temp_caption_path = '{}/{}/{}/{}'.format("TAGAN_data", "text_c10", class_name, caption)
                            print(temp_caption_path)
                            output[temp_caption_path] = get_word_embedding(caption_list)

                            f2.close()
            i += 1

    f.close()
    f3 = open(folder + "caption_embedding" + str(folder_num) + ".pkl","wb")
    pickle.dump(output,f3)

    return output


def get_word_embedding(caption_list):
    #Need to have the length of the description for something?->add later when necessary
    #do we want single tensor for entire sentence or list of tensors for each word?

    output = []
    for caption in caption_list:
        temp_caption = caption.split()
        temp_caption[len(temp_caption)-1] = temp_caption[len(temp_caption)-1].rstrip(".\n")
        #Tensor of list of word vectors
        word_vecs = torch.Tensor([word_embedding[w.lower()] for w in temp_caption])
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

print("Loading fasttext model...")
word_embedding = fasttext.load_model(FT_file)
print("Fast text is loaded!")
output = load_dataset(img_files, caption_files, classes_file)
