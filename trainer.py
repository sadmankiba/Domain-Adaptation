import numpy as np
import torch
from tqdm import tqdm
import cv2
import os
import tensorboard
import matplotlib.pyplot as plt
import json

import losses
#import conventional_dataloader as dataloader
import dataloader as dataloader

import Model_Zoo
import metrics

# TORCH IMPORTS
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import torch.nn as F
from tensorboard import program
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





def train_model(config):

    ''' Trains the model based on config file specifications for segmentation'''


    # BASIC HYPER PARAMETERS
    epochs = config["epochs"]
    batch_size = config["batch_size"]
    tile_size = config["tile_size"]
    lr = config["lr"]
    device = torch.device(f"cuda:{config['device_id']}")
    best_loss = 1000
    best_epoch = 0
    best_dice = 0
    patience_count = 0
    classes = config["classes"]


    # OUPUT DIRECTORY
    config["output_dir"] = os.path.join(config["output_dir"], config["experiment_name"])
    os.makedirs(config["output_dir"], exist_ok= True) 

    # DUMPY JSON
    with open(os.path.join(config["output_dir"], "config.json"), 'w') as f:
        save_dict = json.dumps(config, indent=2)
        f.write(save_dict)



    # MODEL
    model = Model_Zoo.get_model(model_architecture = config["model_architecture"], 
                                backbone = config["backbone"], 
                                classes = config["classes"], 
                                device_id = config["device_id"])
    
    if config["retrain_path"] is not None:
        print("Retraining Mode")
        model = torch.load(config["retrain_path"])
    

    model.to(device)
    model.__setattr__("config", config)


    # LOSS FUNCTION
    loss_function = losses.get_loss_function(config["loss"])

    
    # OPTIMIZER
    optimizer = optim.Adam( model.parameters(), lr=lr)

    
    # LR SCHEDULAR
    scheduler = StepLR(optimizer, step_size=60, gamma=0.1)



    # DATALOADER
    augmentation_config = config["augmentations"]
    train_fetcher = dataloader.Dataloader(split = 'train', augmentations = augmentation_config, data_dir = config["training_dir"], preprocessing_config=config["preprocessing_config"])
    val_fetcher = dataloader.Dataloader(split = 'val', augmentations = None, data_dir = config["training_dir"], preprocessing_config=config["preprocessing_config"])
    train_dataloader = DataLoader(train_fetcher, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_fetcher, batch_size=batch_size, shuffle=False)



    # TENSOR BOARD
    os.makedirs(os.path.join(config["output_dir"], "train"), exist_ok=True)
    os.makedirs(os.path.join(config["output_dir"], "val"), exist_ok=True)
    writer_train = SummaryWriter(log_dir=os.path.join(config["output_dir"], "train"))
    writer_val =SummaryWriter(log_dir=os.path.join(config["output_dir"], "val"))
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', config["output_dir"]])
    url = tb.launch()
    print(f"Tensorflow Link {url}\n")
    

    # METRICS
    dice_metric = metrics.get_metrics("dice_score", classes = config["classes"])
    softmax = F.Softmax(dim=1)


    # TRAINING LOOP
    
    for epoch in range(1, epochs+1):


        #===========================================================
                            # TRAINING LOOP
        #=========================================================== 
        print(f"Epoch: {epoch}")
        batch_iter = tqdm(enumerate(train_dataloader),"Training",total=len(train_dataloader),disable=False)
        train_loss_list = []
        model.train()

        # Intermidiate outputs
        # os.makedirs("/home/arshadk/Projects/Code_base/Torch_Training_Script/Debug/" +str(epoch), exist_ok= True )
        # save_dir = "/home/arshadk/Projects/Code_base/Torch_Training_Script/Debug/" +str(epoch) + "/"

        for _, (x, y) in batch_iter:
            
            # Push img and labels on gpu
            x = x.to(device)
            y = y.to(device)

            # Set Optimizer gradients to zero
            optimizer.zero_grad()

            # Forward Pass
            pred_output = model(x)

            # Loss Calculation
            loss = loss_function(pred_output, y)

            # Back propogation
            loss.backward()

            # Update Weights
            optimizer.step()

            # Collect loss value
            train_loss_list.append(loss.item())


            # Metric Calculation
            pred_output_copy = softmax(pred_output.detach().clone()).argmax(1)
            classwise_train_dice_score = dice_metric.calculate_batch_dice(pred_output_copy, y)

            # Update Loss on Progress Bar
            train_progress_bar_dict = {"Loss":loss.item()}
            for class_index in range(classes):
                train_progress_bar_dict[f"D_c{class_index}"] = classwise_train_dice_score[class_index].item()
            batch_iter.set_postfix(train_progress_bar_dict)

        # Print mean loss and dice
        mean_train_loss = np.mean(train_loss_list)
        classwise_global_train_dice = dice_metric.calculate_global_dice()
        dice_metric.reset()
        print(f"Train Loss: {mean_train_loss:.3f}")
        for class_index in range(classes):
            print(f"Dice class {class_index}: {classwise_global_train_dice[class_index]:.3f}")

        scheduler.step()
        


        #===========================================================
                            # VALIDATION
        #=========================================================== 



        val_batch_iter = tqdm(enumerate(val_dataloader),"Validation",total=len(val_dataloader),disable=False)
        val_loss_list = []
        model.eval()

        for _, (x, y) in val_batch_iter:
            
            # Push on gpu
            x = x.to(device)
            y = y.to(device)

            
            # Inference
            with torch.no_grad():
                pred_output = model(x)

            
            # Loss Calculation
            loss = loss_function(pred_output, y)
            val_loss_list.append(loss.item())


            # Classwise Dice Calculation
            pred_output_copy = softmax(pred_output.detach().clone()).argmax(1)
            classwise_val_dice_score = dice_metric.calculate_batch_dice(pred_output_copy, y)



            # Update Loss on Progress Bar
            progress_bar_dict = {'Val Loss':loss.item()}
            for class_index in range(classes):
                progress_bar_dict[f"D_c{class_index}"] = classwise_val_dice_score[class_index].item()
            val_batch_iter.set_postfix(progress_bar_dict)


        # Print Mean Loss and Dice
        mean_val_loss = np.mean(val_loss_list)
        classwsie_global_val_dice = dice_metric.calculate_global_dice()
        dice_metric.reset()
        print(f"Val Loss: {mean_val_loss:.3f}")
        for class_index in range(classes):
            print(f"Dice class {class_index}: {classwsie_global_val_dice[class_index]:.3f}")

    


        #===========================================================
                            # MODEL SELECTION
        #===========================================================    

        if mean_val_loss < best_loss:
            # save_path = os.path.join(config["output_dir"], "best_model_{}.model".format(config['model_name']))
            # torch.save(model, save_path)
            best_loss = mean_val_loss
            best_epoch = epoch
            best_dice = np.mean(classwsie_global_val_dice)
            patience_count = 0
            print(f"\nNew model saved with Loss {mean_val_loss:.3f} and Dice {best_dice:.3f} at epoch {epoch}")
            save_path = os.path.join(config["output_dir"], "best_model_epoch_{}_loss_{}.model".format(str(epoch), str(round(mean_val_loss, 3))))
            torch.save(model, save_path)
        
            

        else:
            patience_count +=1
            print(f"\nlast model save with Loss {best_loss} and Dice {best_dice:.3f} at epoch {best_epoch}")

        # SAVE ALL MODELS
        #save_path = os.path.join(config["output_dir"], "epoch_{}_loss_{}.model".format(str(epoch), str(round(mean_val_loss, 3))))
        #torch.save(model, save_path)
            
        if patience_count > config["patience"]:
            print("Early Stopping")
            break



        print("\n=====================================================\n")



        #===========================================================
                            # LOGGING INFORMATION
        #===========================================================    

        writer_train.add_scalar("Learning Rate", optimizer.param_groups[0]['lr'], epoch)
        writer_train.add_scalar("Loss", mean_train_loss, epoch)
        writer_val.add_scalar("Loss", mean_val_loss, epoch)
        
        for class_index in range(classes):
            writer_train.add_scalar(f"Dice Class {class_index}", classwise_global_train_dice[class_index], epoch)
            writer_val.add_scalar(f"Dice Class {class_index}", classwsie_global_val_dice[class_index], epoch)




    writer_train.close()



        

        

        

        



    



    

    
    



    

    
    



    return 0

