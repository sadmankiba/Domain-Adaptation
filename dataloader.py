import os
import cv2
import torch
#from augmentations import augment_data
import numpy as np
from IP_Tools.commons import GlobalPreprocess
import torchvision.transforms as T


class Dataloader():

    def __init__(self, split, augmentations, data_dir, preprocessing_config):

        self.data_dir = data_dir
        self.image_dir = os.path.join(data_dir, split, split + "_data_all")
        self.label_dir = os.path.join(data_dir, split, split + "_data_all", "Label")
        self.augmentations = augmentations
        self.split = split
        self.mean = preprocessing_config["mean"]
        self.std_dev = preprocessing_config["std_dev"]
        self.preprocessor = T.Compose([T.ToTensor(), T.Normalize(mean=self.mean, std=self.std_dev)])
        
        self.image_name_list = [image_name for image_name in os.listdir(self.image_dir) if image_name.endswith(".png")]

    def __len__(self):
        return len(self.image_name_list)


    def __getitem__(self, index):

        # Read Image
        img = cv2.imread(os.path.join(self.image_dir, self.image_name_list[index]))
        label = cv2.imread(os.path.join(self.label_dir, self.image_name_list[index].replace(".png", ".png")))
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)

        # # FOR MNC
        # img = img[0:512, 0:512, :]
        # label = label[0:512, 0:512]
        

        # Preprocessor for color coding 
        # label[label == 154] = 1
        # label[label == 255] = 2
        # label[label == 92] = 3
        # label[label == 80] = 0

        label[label == 255] = 1
        
        # Augmentations
        if self.augmentations is not None:
            img, label = augment_data(img, label, self.augmentations)

            if self.augmentations["save_augmentations"]:
                image_save_path = os.path.join(self.augmentations["save_path"], self.image_name_list[index])        
                label_save_path = os.path.join(self.augmentations["save_path"], "Label", self.image_name_list[index].replace(".png", "_modlabel.png"))
                os.makedirs(os.path.join(self.augmentations["save_path"], "Label"), exist_ok= True)

                cv2.imwrite(image_save_path, img)
                cv2.imwrite(label_save_path, label)


                
        # # Torch conversion
        img = self.preprocessor(img)
        label = torch.tensor(label).long()

    
        return img, label
    

def color_encoding(label):

    label[label == 154] = 1
    label[label == 255] = 2
    label[label == 92] = 3
    
    return label