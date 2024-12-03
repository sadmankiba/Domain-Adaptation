import os
import cv2
import torch
from augmentations import augment_data, Augmentor
import numpy as np
import torchvision.transforms as T
import random

# Seeds
# Set a seed for torch, numpy, and Python's random module for full reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# For GPU operations, set the seed for CUDA as well
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Dataloader():

    def __init__(self, split, augmentations, data_dir, preprocessing_config):

        self.data_dir = data_dir
        self.image_name_list, self.label_list = get_city_scape_data(split, data_dir)
        self.augmentations = augmentations
        self.split = split
        self.mean = preprocessing_config["mean"]
        self.std_dev = preprocessing_config["std_dev"]
        self.preprocessor = T.Compose([T.ToTensor(), T.Normalize(mean=self.mean, std=self.std_dev)])
        self.augmetor = Augmentor()
        #self.image_name_list = [image_name for image_name in os.listdir(self.image_dir) if image_name.endswith(".png")]
        print("Total Data", len(self.image_name_list))

    def __len__(self):
        return len(self.image_name_list)


    def __getitem__(self, index):
        """
        
        Returns:
            img: Tensor. Shape: (3, 512, 1024). Float Tensor
            label: Tensor. Shape: (512, 1024). Long Tensor
        """
        # Read Image
        img = cv2.imread(self.image_name_list[index])
        label = cv2.imread(self.label_list[index])
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)

        label = color_encoding(label)

        # Reduce the Size
        img = cv2.resize(img, (1024, 512), interpolation=cv2.INTER_CUBIC)
        label = cv2.resize(label, (1024, 512), interpolation=cv2.INTER_LINEAR_EXACT)

        
        
        # Augmentations
        if self.augmentations is not None:

            img, label = self.augmetor.augment_image(image = img, label = label)
            
            # if self.augmentations["save_augmentations"]:
            #     image_save_path = os.path.join(self.augmentations["save_path"], self.image_name_list[index])        
            #     label_save_path = os.path.join(self.augmentations["save_path"], "Label", self.image_name_list[index].replace(".png", "_modlabel.png"))
            #     os.makedirs(os.path.join(self.augmentations["save_path"], "Label"), exist_ok= True)

            #     cv2.imwrite(image_save_path, img)
            #     cv2.imwrite(label_save_path, label)


                
        # # Torch conversion
        img = self.preprocessor(img)
        label = torch.tensor(label).long()

    
        return img, label
    

def color_encoding(label):
    label[label == 1] = 0
    label[label == 2] = 0
    label[label == 3] = 0
    label[label == 4] = 0
    label[label == 5] = 0
    label[label == 6] = 0
    label[label == 7] = 1
    label[label == 8] = 2
    label[label == 9] = 0
    label[label == 10] = 0
    label[label == 11] = 3
    label[label == 12] = 4
    label[label == 13] = 5
    label[label == 14] = 0
    label[label == 15] = 0
    label[label == 16] = 0
    label[label == 17] = 6
    label[label == 18] = 0
    label[label == 19] = 7
    label[label == 20] = 8
    label[label == 21] = 9
    label[label == 22] = 10
    label[label == 23] = 11
    label[label == 24] = 12
    label[label == 25] = 13
    label[label == 26] = 14
    label[label == 27] = 15
    label[label == 28] = 16
    label[label == 29] = 0
    label[label == 30] = 0
    label[label == 31] = 17
    label[label == 32] = 18
    label[label == 33] = 19

    return label
    
    
    
    

    return label


def get_city_scape_data(split, data_dir):

    image_dir = os.path.join(data_dir, "leftImg8bit")
    label_dir = os.path.join(data_dir, "gtFine")

    image_dir_path_list = []
    label_dir_path_list = []

    image_dir_path = os.path.join(image_dir, split)
    label_dir_path = os.path.join(label_dir, split)
    i = 0

    for root, dirs, files in os.walk(image_dir_path):
        for image_file in files:

            if image_file.lower().endswith(".png"):
                city_name = os.path.basename(root)
                image_path = os.path.join(root, image_file)
                label_file = image_file.replace("_leftImg8bit.png", "_gtFine_labelIds.png")
                label_path = os.path.join(label_dir_path, city_name, label_file)

                # Append data
                image_dir_path_list.append(image_path)
                label_dir_path_list.append(label_path)

    return image_dir_path_list, label_dir_path_list
