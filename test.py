import torch
from dataloader import Dataloader
import dataloader
from torch.utils.data import DataLoader
from tqdm import tqdm
from metrics import ImageMetrics
import numpy as np

# Model Path
DATA_PATH = "/users/Sadman/Test_Time_Domain_Adaptation/data/"
MODEL_PATH = "/users/Sadman/Test_Time_Domain_Adaptation/Experiments/Baseline for Cityscape with 20 classes with Basic Augmentation/best_model_epoch_17_loss_0.388.model"

# Batchsize
batch_size = 8


# Loading Model
model = torch.load(MODEL_PATH)

# Dataloader
val_fetcher = dataloader.Dataloader(split = 'val', augmentations = None, data_dir = DATA_PATH, preprocessing_config={"divideBy":255, "mean":[0, 0, 0], "std_dev": [1,1, 1]})
val_dataloader = DataLoader(val_fetcher, batch_size=batch_size, shuffle=False)

# Metrics
dice_metric = ImageMetrics(classes = 20)




model.train()
for k in tqdm(range(3)):

    if k < 2: 
        for x,y in val_dataloader:
            x = x.cuda()
            with torch.no_grad():
                output = model(x)

    else:
        print("Moved to eval Mode")
        model.eval()
        for x, y in val_dataloader:
            x = x.cuda()
            y = y.cuda()
            with torch.no_grad():
                pred_output = model(x)
        
            pred_output_copy = torch.softmax(pred_output.detach().clone(), dim = 1).argmax(1)
            classwise_val_dice_score = dice_metric.fit(pred_output_copy, y)
        
        classwsie_global_val_dice = dice_metric.get_global_dice()
        print("Mean Dice:", np.mean(classwsie_global_val_dice))


    

