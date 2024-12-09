import torch
from dataloader import Dataloader
import dataloader
from torch.utils.data import DataLoader
from tqdm import tqdm
from metrics import ImageMetrics
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Model Path
DATA_PATH = "/users/Sadman/Test_Time_Domain_Adaptation/data/"
MODEL_PATH = "/users/Sadman/Test_Time_Domain_Adaptation/Experiments/Baseline for Cityscape with 20 classes with Basic Augmentation/best_model_epoch_17_loss_0.388.model"
# MODEL_PATH = "/users/Sadman/Test_Time_Domain_Adaptation/Experiments/Baseline for Cityscape with 20 classes/epoch_17_loss_0.412.model"

# Batchsize
batch_size = 8
dice_metric = ImageMetrics(classes = 20)







def run_test_time_bn_with_test_data(val_dataloader):
    # Loading Model
    model = torch.load(MODEL_PATH)
    
    model.train()
    for k in range(3):

        if k < 2: 
            for x,y in tqdm(val_dataloader):
                x = x.cuda()
                with torch.no_grad():
                    output = model(x)

        else:
            print("Moved to eval Mode")
            model.eval()
            for x, y in tqdm(val_dataloader):
                x = x.cuda()
                y = y.cuda()
                with torch.no_grad():
                    pred_output = model(x)
            
                pred_output_copy = torch.softmax(pred_output.detach().clone(), dim = 1).argmax(1)
                classwise_val_dice_score = dice_metric.fit(pred_output_copy, y)
            
            classwsie_global_val_dice = dice_metric.get_global_dice()
            print("Mean Dice:", np.mean(classwsie_global_val_dice))

       
def run_test_time_bn_with_test_train_mixed_data(train_dataloader, val_dataloader, mix=False):

    # Loading Model
    model = torch.load(MODEL_PATH)

    model.train()
    total_iters = 3
    train_data_iters = [0, 1]
    eval_iters = [2]
    for k in range(total_iters):
        if k in train_data_iters:
            for x, y in tqdm(train_dataloader):
                x = x.cuda()
                with torch.no_grad():
                    output = model(x)
        else:
            print("Moved to eval Mode")
            model.eval()
            for x, y in tqdm(val_dataloader):
                x = x.cuda()
                y = y.cuda()
                with torch.no_grad():
                    pred_output = model(x)
            
                pred_output_copy = torch.softmax(pred_output.detach().clone(), dim = 1).argmax(1)
                classwise_val_dice_score = dice_metric.fit(pred_output_copy, y)
            
            classwsie_global_val_dice = dice_metric.get_global_dice()
            print("Mean Dice:", np.mean(classwsie_global_val_dice))
    
    output_path = Path(MODEL_PATH).parent / f"ttn_{mix}.pth"
    torch.save(model, output_path)

    return np.mean(classwsie_global_val_dice)

def run_mixed_with_varying_batch_size():
    # Dataloader

    for i in tqdm(range(1, 24)):

        j = i/10
        batch_size = i

        score_list = []

        x_list = []

        val_fetcher = dataloader.Dataloader(split = 'val', augmentations = None, data_dir = DATA_PATH, 
                        preprocessing_config={"divideBy":255, "mean":[0, 0, 0], "std_dev": [1,1, 1]}, mix=False)
        val_dataloader = DataLoader(val_fetcher, batch_size=batch_size, shuffle=True)

        train_fetcher = dataloader.Dataloader(split = 'val', augmentations = None, data_dir = DATA_PATH, 
                        preprocessing_config={"divideBy":255, "mean":[0, 0, 0], "std_dev": [1,1, 1]}, mix=False)
        train_dataloader = DataLoader(train_fetcher, batch_size=batch_size, shuffle=True)

        # Metrics
        dice_metric = ImageMetrics(classes = 20)

        score = run_test_time_bn_with_test_train_mixed_data(train_dataloader = train_dataloader, val_dataloader = val_dataloader)

        score_list.append(score)
        x_list.append(j)

    plt.plot(x_list, score_list)
    plt.savefig("alpha_vs_accuraacy.png")


if __name__ == "__main__":
    val_fetcher = dataloader.Dataloader(split = 'val', augmentations = None, data_dir = DATA_PATH, 
                        preprocessing_config={"divideBy":255, "mean":[0, 0, 0], "std_dev": [1,1, 1]}, mix=False)
    val_dataloader = DataLoader(val_fetcher, batch_size=batch_size, shuffle=True)
    
    train_mix=0.2
    train_fetcher = dataloader.Dataloader(split = 'val', augmentations = None, data_dir = DATA_PATH, 
                        preprocessing_config={"divideBy":255, "mean":[0, 0, 0], "std_dev": [1,1, 1]}, mix=train_mix)
    train_dataloader = DataLoader(train_fetcher, batch_size=batch_size, shuffle=True)
    
    # run_test_time_bn_with_test_data(val_dataloader)  
    run_test_time_bn_with_test_train_mixed_data(train_dataloader, val_dataloader, mix=train_mix)


    
    

