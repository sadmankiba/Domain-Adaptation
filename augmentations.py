import numpy as np
import cv2
import os
import albumentations as A
#from skimage import color


# HED STAIN AUGMENTATION
def hed_augmentation(img, theta=0.02):
    threshold = 0.9
    alpha = np.random.uniform(1 - theta, 1 + theta, (1, 3))
    beta = np.random.uniform(-theta, theta, (1, 3))
    img = np.array(img)
    gray_img = color.rgb2gray((img))
    background = (
        gray_img > threshold
    )  
    # * (gray_img-self.threshold)/(1-self.threshold)
    background = background[:, :, np.newaxis]

    s = color.rgb2hed(img)
    ns = alpha * s + beta  # perturbations on HED color space
    nimg = color.hed2rgb(ns)

    imin = nimg.min()
    imax = nimg.max()
    rsimg = (nimg - imin) / (imax - imin)  # rescale
    rsimg = (1 - background) * rsimg + background * img / 255
    rsimg = (255 * rsimg).astype("uint8")

    return rsimg

class Augmentor():
    
    def __init__(self):
        
        augmentation_list = [A.HorizontalFlip(p=0.3),  
                             A.Rotate(limit=30, p=0.2, border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0)),
                             A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.2, p=0.3),
                             A.RandomGamma(gamma_limit=(80, 120), p=0.5)]

        self.augmentor = A.Compose(augmentation_list)
        
    def augment_image(self, image, label):

        augmented_data = self.augmentor(image=image, mask=label)
        augmented_image = augmented_data["image"]
        augmented_label = augmented_data["mask"]

        return augmented_image, augmented_label
    



    
def augment_data(img, label, augmentations):
    

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Geometric  Augmentations
    if "geometric" in augmentations:

        if "rotation" in augmentations["geometric"]:
            augmentor = A.Compose([A.VerticalFlip(p=0.5),A.HorizontalFlip(p=0.5),  A.RandomRotate90(p=0.5)])
            rotation = augmentor(image = img, mask = label)
            img = rotation["image"]
            label = rotation["mask"]

        
        if "zoom" in augmentations["geometric"]:
            augmentor = A.Affine (scale=(0.7, 1.3),keep_ratio = True, p=augmentations["geometric"]["zoom"])
            zoom = augmentor(image = img, mask = label)
            img = zoom["image"]
            label = zoom["mask"]

        if "jpeg" in augmentations["geometric"]:
            augmentor = A.ImageCompression(quality_lower=99, quality_upper=100, always_apply=False, p=augmentations["geometric"]["jpeg"]) 
            compression = augmentor(image = img, mask = label)
            img = zoom['image']
            label = compression['mask']


    # Color Augmentations
    if "color" in augmentations:
        
        # Pobabilities
        color_augmentation_list = list(augmentations["color"].keys())
        color_augmentation_list.append("none")
        augmentation__probability =list(augmentations["color"].values())
        none_propability = 1 - sum(augmentation__probability)
        augmentation__probability.append(none_propability)


        random_augmentation = np.random.choice(a = color_augmentation_list, size = 1, p=augmentation__probability)[0]


        if random_augmentation == "contrast_brightness_gamma":
            augmentor = A.Compose([A.RandomBrightnessContrast(p=1),A.RandomGamma(p=1)])
            brightness_gamma_contrast = augmentor(image = img, mask = label)                 
            img = brightness_gamma_contrast["image"]
            label = brightness_gamma_contrast["mask"]


        elif random_augmentation == "hed":
            img = hed_augmentation(img)

        elif random_augmentation == "none":
            pass

        else:
            assert("Augmentation not found")

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img, label