import os 

data_dir = "/users/Sadman/Test_Time_Domain_Adaptation/data/"
leftImg8bit_dir = os.path.join(data_dir, "leftImg8bit")
gtFine_dir = os.path.join(data_dir, "gtFine")
foggy_dir = os.path.join(data_dir, "leftImg8bit_foggyDBF_short")
output_dir = "/users/Sadman/Test_Time_Domain_Adaptation/outputs/"
city = "frankfurt"
fog_score = "0.02"


without_aug_config = {
    # Paths

    # Normal Data
    # "output_dir": f"{output_dir}/outputs_without_augmentations_munster/", 
    # "training_data": f"{leftImg8bit_dir}/val/munster/",
    # "gt_path": f"{gtFine_dir}/val/munster/",

    # # Foggy Data
    "fog_score": fog_score,
    "output_dir": f"{output_dir}/outputs_without_augmentations_munster_foggy_{fog_score}/", 
    "training_data": f"{foggy_dir}/val/munster/",
    "gt_path": f"{gtFine_dir}/val/munster/",

    # Model Parameters
    "model_path": "/users/Sadman/Test_Time_Domain_Adaptation/Experiments/Baseline for Cityscape with 20 classes/epoch_17_loss_0.412.model",
    "model_architecture": "Unet",
    "backbone":"resnet50",
    "preprocessing_config": {"divideBy":255, "mean":[0, 0, 0], "std_dev": [1,1, 1]},
    #"preprocessing_config": {"divideBy":255, "mean":[0.406, 0.456, 0.485], "std_dev":[0.225, 0.224, 0.229]},
    #"preprocessing_config": {"divideBy":255, "mean":[ 0.695, 0.470, 0.704], "std_dev":[ 0.119 , 0.190,0.138]},
    "preprocessing_type": "global",
    "model_weights":False,   # if False, load model from model_path, otherwise use pretrained model

    # Device
    "classes":20,

    "tile_size": 1024,
    "device_id":0,
}

with_aug_config = {


    # Normal Data
    # "output_dir": "/users/Sadman/Test_Time_Domain_Adaptation/outputs/outputs_with_augmentations_munster/", 
    # "training_data":"/users/Sadman/Test_Time_Domain_Adaptation/data/leftImg8bit/val/munster/",
    # "gt_path": "/users/Sadman/Test_Time_Domain_Adaptation/data/gtFine/val/munster/",

    # # Foggy Data
    "fog_score": fog_score,
    "output_dir": f"{output_dir}/outputs_with_augmentations_{city}_foggy_{fog_score}/", 
    "training_data": f"{foggy_dir}/val/{city}/",
    "gt_path": f"{gtFine_dir}/val/{city}/",


    # Model Parameters
    "model_path": "/users/Sadman/Test_Time_Domain_Adaptation/Experiments/Baseline for Cityscape with 20 classes with Basic Augmentation/best_model_epoch_17_loss_0.388.model",
    "model_architecture": "Unet",
    "backbone":"resnet50",
    "preprocessing_config": {"divideBy":255, "mean":[0, 0, 0], "std_dev": [1,1, 1]},
    "preprocessing_type": "global",
    "model_weights":False,

    # Device
    "classes":20,

    "tile_size": 1024,
    "device_id":0,
}


config = with_aug_config