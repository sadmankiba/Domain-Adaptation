from augmentations import Augmentor
import cv2

if __name__  == "__main__":


    img = cv2.imread("../data/leftImg8bit/train/aachen/aachen_000035_000019_leftImg8bit.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    label = cv2.imread("../data/gtFine/train/aachen/aachen_000035_000019_gtFine_labelIds.png")
    augmentor = Augmentor()
    cv2.imwrite("./Augmentations/img.png", img)

    for i in range(20):

        img, label = augmentor.augment_image(image=img, label = label)
        cv2.imwrite(f"./Augmentations/{i}.png", img)
        cv2.imwrite(f"./Augmentations/{i}_mask.png", label)
        
    





    




