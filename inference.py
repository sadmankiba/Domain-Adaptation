import torch
import Model_Zoo
from glob import glob
import torchvision.transforms as T
import cv2
import os
from tqdm import tqdm
from metrics import ImageMetrics
from dataloader import color_encoding

def infer_model(model_config, data_path, output_path =None, gt_path = None, fog_score = None):

    model_architecture = model_config["model_architecture"]
    backbone = model_config["backbone"]
    classes = model_config["classes"]
    device = torch.device(model_config["device_id"])
    preprocessing_config = model_config["preprocessing_config"]
    mean = preprocessing_config["mean"]
    std_dev = preprocessing_config["std_dev"]
    preprocessor = T.Compose([T.ToTensor(), T.Normalize(mean=mean, std=std_dev)])
    classes = model_config["classes"]
    grlvl = int(255/(classes -1))

    # Model
    if not model_config["model_weights"]:
        model = torch.load(model_config["model_path"])

    else:
        model = Model_Zoo.get_model(model_architecture, backbone, classes, model_config["device_id"])
        model_weights = torch.load(model_config["model_path"])
        model.load_state_dict(model_weights)

    model.eval()
    model.to(device)
    print("loaded model")

    # Metrics
    metric_object = ImageMetrics(classes = classes)
    
    image_paths = None
    if 'foggy' in data_path:
        image_paths = glob(data_path + f"*{fog_score}.png")
    else:
        image_paths = glob(data_path + "*.png")

    for image_path in tqdm(image_paths):
   

        # READ IMAGE
        image_name = os.path.basename(image_path)
        img = cv2.imread(image_path)

        img = cv2.resize(img, (1024, 512), interpolation=cv2.INTER_CUBIC)
        

    
        img = preprocessor(img)
        img = torch.unsqueeze(img, 0)

        # hamburg_000000_002338_leftImg8bit_foggy_beta_0.01.png
        # hamburg_000000_002338_gtFine_labelIds
        # READ GT
        if gt_path is not None:
            label_name = os.path.basename(image_path)
            image_path.find("leftImg8bit")
            if 'foggy' in data_path:
                label_path = os.path.join(gt_path, label_name.replace(f"leftImg8bit_foggy_beta_{fog_score}", "gtFine_labelIds"))
            else:
                label_path = os.path.join(gt_path, label_name.replace("leftImg8bit", "gtFine_labelIds"))
            label = cv2.imread(label_path)
            label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
            label = cv2.resize(label, (1024, 512), interpolation=cv2.INTER_LINEAR_EXACT)
            label = color_encoding(label)
            label = torch.tensor(label).long()
            label.to(device)
            label =torch.unsqueeze(label, 0)
            label = label.to(device)

        # INFERENCE
        with torch.no_grad():
            img = img.to(device)
            output = model(img)
            output = torch.softmax(output, dim = 1).argmax(1)

            # CALCULATE METRICS
            if gt_path is not None:
                metric_object.fit(output, label)
        
        # WRITE OUTPUT
        if output_path is not None:
            output = output.cpu().numpy()[0]*grlvl
            cv2.imwrite(output_path + image_name[:-4] + ".png",output)
    
    # CALCULATE & PRINT DICE
    if gt_path is not None:
        dice = metric_object.get_global_dice()
        precision = metric_object.get_global_precision()
        recall = metric_object.get_global_recall()

        output_str = ""
        for class_index in range(classes):
            output_str += f"Class: {class_index}\n"
            output_str += f"Dice:{dice[class_index]:.2f}\n"
            output_str += f"Recall:{recall[class_index]:.2f}\n"
            output_str += f"Precision:{precision[class_index]}\n"
            
        output_str += f"Global Dice: {sum(dice)/classes:.2f}"
        print(output_str)
        with open(output_path + "results.txt", "w") as f:
            f.write(output_str)        
